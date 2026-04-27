/*
 * Job-Shop Scheduling Problem - Parallel Shifting Bottleneck (OpenMP)
 * Computação de Alto Desempenho - MEI IPCA 2025/2026
 *
 * ─── Foster Methodology ─────────────────────────────────────────────────────
 * PARTITION   : The evaluation of each unscheduled machine (computing its
 *               1-machine Cmax) is an independent task — machines share the
 *               same read-only graph but write to independent output slots.
 * COMMUNICATION: Threads share the disjunctive graph (read-only during the
 *               parallel evaluation phase). Writing to the graph (fixing a
 *               machine) is done sequentially after the parallel phase.
 *               The only shared write target during parallel execution is
 *               the per-machine Cmax result array, which uses independent
 *               slots — no locking required there.
 * AGGLOMERATION: Each thread evaluates one machine per iteration of the
 *               #pragma omp parallel for. The 1-machine solver (Schrage)
 *               is self-contained and uses only thread-local arrays.
 * MAPPING      : Static scheduling (schedule(static)) — each machine takes
 *               roughly the same amount of work (O(n log n)), so static
 *               distribution is fair.
 *
 * Shared read-only (during parallel region):
 *   r_time[], q_time[]            — release/tail times (recomputed before
 *                                    each parallel region, never written
 *                                    inside it)
 *   ops_on_machine[][], proc_time[][], machine_id[][]
 *
 * Shared read-write:
 *   cmax_result[]   — each thread writes to its own machine slot, no race
 *   seq_result[][]  — each thread writes to its own machine row, no race
 *   adj/pred graph  — written ONLY in the sequential fix/reoptimise phase
 *
 * Thread-local variables (declared inside parallel region):
 *   r[], p[], q[], seq[]  — local copies used by Schrage solver
 *
 * Critical sections: NONE during the parallel bottleneck evaluation.
 * The only synchronisation is the implicit barrier at the end of
 * #pragma omp parallel for, after which the master selects the bottleneck
 * and fixes the graph sequentially.
 *
 * Mutual exclusion technique: none required for the core parallel region.
 * OpenMP barrier (implicit at end of parallel for) ensures all threads
 * have written their cmax_result before the master reads them.
 *
 * Usage: jobshop_par <input> <output> <num_threads>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

#define MAX_JOBS 30
#define MAX_MACHINES 30
#define MAX_OPS (MAX_JOBS * MAX_MACHINES)
#define MAX_ADJ 64
#define INF 0x3fffffff

/* ── Input ──────────────────────────────────────────────────── */
static int num_jobs, num_machines, total_ops;
static int machine_id[MAX_JOBS][MAX_MACHINES];
static int proc_time[MAX_JOBS][MAX_MACHINES];

#define SRC (total_ops)
#define SNK (total_ops + 1)
#define N (total_ops + 2)

/* ── Disjunctive graph ──────────────────────────────────────── */
static int adj[MAX_OPS + 2][MAX_ADJ];
static int adj_w[MAX_OPS + 2][MAX_ADJ];
static int adj_cnt[MAX_OPS + 2];
static int pred[MAX_OPS + 2][MAX_ADJ];
static int pred_w[MAX_OPS + 2][MAX_ADJ];
static int pred_cnt[MAX_OPS + 2];

static void add_arc(int from_node, int to_node, int weight)
{
    adj[from_node][adj_cnt[from_node]] = to_node;
    adj_w[from_node][adj_cnt[from_node]] = weight;
    adj_cnt[from_node]++;
    pred[to_node][pred_cnt[to_node]] = from_node;
    pred_w[to_node][pred_cnt[to_node]] = weight;
    pred_cnt[to_node]++;
}
static void remove_arc(int from_node, int to_node)
{
    for (int edge_index = 0; edge_index < adj_cnt[from_node]; edge_index++)
    {
        if (adj[from_node][edge_index] == to_node)
        {
            adj[from_node][edge_index] = adj[from_node][adj_cnt[from_node] - 1];
            adj_w[from_node][edge_index] = adj_w[from_node][adj_cnt[from_node] - 1];
            adj_cnt[from_node]--;
            break;
        }
    }
    for (int edge_index = 0; edge_index < pred_cnt[to_node]; edge_index++)
    {
        if (pred[to_node][edge_index] == from_node)
        {
            pred[to_node][edge_index] = pred[to_node][pred_cnt[to_node] - 1];
            pred_w[to_node][edge_index] = pred_w[to_node][pred_cnt[to_node] - 1];
            pred_cnt[to_node]--;
            break;
        }
    }
}

/* ── Release / tail times (computed sequentially) ───────────── */
/* These arrays are written ONLY in compute_release/compute_tails,
   which are called BEFORE any parallel region — so they are
   read-only from the threads' perspective. */
static int r_time[MAX_OPS + 2];
static int q_time[MAX_OPS + 2];

static void compute_release(void)
{
    for (int node = 0; node < N; node++)
        r_time[node] = 0;
    for (int pass = 0; pass < N; pass++)
    {
        int changed = 0;
        for (int from_node = 0; from_node < N; from_node++)
            for (int edge_index = 0; edge_index < adj_cnt[from_node]; edge_index++)
            {
                int to_node = adj[from_node][edge_index];
                int edge_weight = adj_w[from_node][edge_index];
                if (r_time[from_node] + edge_weight > r_time[to_node])
                {
                    r_time[to_node] = r_time[from_node] + edge_weight;
                    changed = 1;
                }
            }
        if (!changed)
            break;
    }
}
static void compute_tails(void)
{
    for (int node = 0; node < N; node++)
        q_time[node] = 0;
    for (int pass = 0; pass < N; pass++)
    {
        int changed = 0;
        for (int to_node = 0; to_node < N; to_node++)
            for (int edge_index = 0; edge_index < pred_cnt[to_node]; edge_index++)
            {
                int from_node = pred[to_node][edge_index];
                int edge_weight = pred_w[to_node][edge_index];
                if (edge_weight + q_time[to_node] > q_time[from_node])
                {
                    q_time[from_node] = edge_weight + q_time[to_node];
                    changed = 1;
                }
            }
        if (!changed)
            break;
    }
}

/* ── Machine membership ─────────────────────────────────────── */
static int ops_on_machine[MAX_MACHINES][MAX_JOBS];
static int ops_on_machine_cnt[MAX_MACHINES];

/* ── Per-machine Cmax results (written by parallel threads) ─── */
/* Each machine m writes to cmax_result[m] and seq_result[m][].
   Since each thread writes to a DIFFERENT machine's slot,
   there is no race condition — no lock needed. */
static int cmax_result[MAX_MACHINES];
static int seq_result[MAX_MACHINES][MAX_JOBS];

/* ── 1-machine Schrage solver (thread-safe, uses only locals) ── */
#define HEAP_MAX MAX_JOBS
typedef struct
{
    int op, q, r, p;
} HItem;

static int schrage(const int *ops, int n,
                   const int *r, const int *p, const int *q,
                   int *seq,
                   HItem *heap)
{ /* heap is caller-provided (thread-local) */
    static __thread int done[MAX_JOBS];
    (void)ops;

    for (int i = 0; i < n; i++)
        done[i] = 0;
    int heap_size = 0;

    int min_release = INF;
    for (int i = 0; i < n; i++)
        if (r[i] < min_release)
            min_release = r[i];
    int current_time = min_release;
    int sequence_index = 0;
    int cmax = 0;

    while (sequence_index < n)
    {
        for (int i = 0; i < n; i++)
        {
            if (!done[i] && r[i] <= current_time)
            {
                HItem item = {i, q[i], r[i], p[i]};
                int insert_index = heap_size++;
                heap[insert_index] = item;
                while (insert_index > 0)
                {
                    int parent_index = (insert_index - 1) / 2;
                    if (heap[parent_index].q < heap[insert_index].q)
                    {
                        HItem tmp = heap[parent_index];
                        heap[parent_index] = heap[insert_index];
                        heap[insert_index] = tmp;
                        insert_index = parent_index;
                    }
                    else
                        break;
                }
                done[i] = 2;
            }
        }
        if (!heap_size)
        {
            int next_release = INF;
            for (int i = 0; i < n; i++)
                if (!done[i] && r[i] < next_release)
                    next_release = r[i];
            current_time = next_release;
            continue;
        }
        HItem current_item = heap[0];
        heap[0] = heap[--heap_size];
        int current_index = 0;
        while (1)
        {
            int left_child = 2 * current_index + 1;
            int right_child = 2 * current_index + 2;
            int best_index = current_index;
            if (left_child < heap_size && heap[left_child].q > heap[best_index].q)
                best_index = left_child;
            if (right_child < heap_size && heap[right_child].q > heap[best_index].q)
                best_index = right_child;
            if (best_index == current_index)
                break;
            HItem tmp = heap[current_index];
            heap[current_index] = heap[best_index];
            heap[best_index] = tmp;
            current_index = best_index;
        }
        seq[sequence_index++] = current_item.op;
        current_time += current_item.p;
        int completion_with_tail = current_time + current_item.q;
        if (completion_with_tail > cmax)
            cmax = completion_with_tail;
    }
    return cmax;
}

/* ── Machine fix state ──────────────────────────────────────── */
static int machine_fixed[MAX_MACHINES];
static int machine_seq[MAX_MACHINES][MAX_JOBS];

static void fix_machine(int machine, const int *sequence, int operation_count)
{
    for (int i = 0; i < operation_count - 1; i++)
    {
        remove_arc(machine_seq[machine][i], machine_seq[machine][i + 1]);
    }
    for (int order_index = 0; order_index < operation_count - 1; order_index++)
    {
        int from_node = ops_on_machine[machine][sequence[order_index]];
        int to_node = ops_on_machine[machine][sequence[order_index + 1]];
        int processing_time = 0;
        for (int job = 0; job < num_jobs; job++)
            for (int op = 0; op < num_machines; op++)
                if (job * num_machines + op == from_node)
                    processing_time = proc_time[job][op];
        add_arc(from_node, to_node, processing_time);
        machine_seq[machine][order_index] = from_node;
        machine_seq[machine][order_index + 1] = to_node;
    }
    machine_fixed[machine] = 1;
}

/* ── Parallel Shifting Bottleneck ───────────────────────────── */
static void shifting_bottleneck_parallel(void)
{
    int remaining_machines = num_machines;

    while (remaining_machines > 0)
    {

        /* ── Compute r/q sequentially (graph may have changed) ── */
        compute_release();
        compute_tails();

/* ── PARALLEL REGION: evaluate each unscheduled machine ──
 * Each thread i processes machine m = its loop iteration.
 * Reads: r_time[], q_time[], ops_on_machine[][], proc_time[][]
 *        (all read-only in this region)
 * Writes: cmax_result[m], seq_result[m][]
 *         (each thread writes to its own machine slot — no race)
 * Thread-local: r[], p[], q[], seq[], heap[]
 */
#pragma omp parallel for schedule(static) default(none)             \
    shared(machine_fixed, ops_on_machine,                           \
               ops_on_machine_cnt, proc_time, machine_id, num_jobs, \
               num_machines, r_time, q_time,                        \
               cmax_result, seq_result)
        for (int m = 0; m < num_machines; m++)
        {
            if (machine_fixed[m])
            {
                cmax_result[m] = -1;
                continue;
            }
            /* Thread-local arrays — stack-allocated, no sharing */
            int r[MAX_JOBS], p[MAX_JOBS], q[MAX_JOBS], seq[MAX_JOBS];
            HItem heap[MAX_JOBS];

            int n = ops_on_machine_cnt[m];
            /* Fill r/p/q from read-only shared arrays */
            for (int i = 0; i < n; i++)
            {
                int op = ops_on_machine[m][i];
                int ptime = 0;
                for (int j = 0; j < num_jobs; j++)
                    for (int o = 0; o < num_machines; o++)
                        if (j * num_machines + o == op)
                            ptime = proc_time[j][o];
                r[i] = r_time[op];
                p[i] = ptime;
                q[i] = q_time[op];
            }
            /* Solve 1-machine subproblem (thread-local computation) */
            cmax_result[m] = schrage(ops_on_machine[m], n,
                                     r, p, q, seq, heap);
            /* Write sequence to per-machine slot (no race) */
            for (int i = 0; i < n; i++)
                seq_result[m][i] = seq[i];
        }
        /* Implicit OpenMP barrier here — all cmax_result[] written */

        /* ── Sequential: select bottleneck & fix ── */
        int bottleneck_machine = -1;
        int best_cmax = -1;
        for (int m = 0; m < num_machines; m++)
            if (cmax_result[m] > best_cmax)
            {
                best_cmax = cmax_result[m];
                bottleneck_machine = m;
            }

        fix_machine(bottleneck_machine, seq_result[bottleneck_machine],
                    ops_on_machine_cnt[bottleneck_machine]);
        remaining_machines--;

        /* ── Re-optimise fixed machines (sequential per fix) ── */
        compute_release();
        compute_tails();
        for (int m = 0; m < num_machines; m++)
        {
            if (!machine_fixed[m] || m == bottleneck_machine)
                continue;
            int n = ops_on_machine_cnt[m];
            int r[MAX_JOBS], p[MAX_JOBS], q[MAX_JOBS], seq[MAX_JOBS];
            HItem heap[MAX_JOBS];
            for (int i = 0; i < n; i++)
            {
                int op = ops_on_machine[m][i], ptime = 0;
                for (int j = 0; j < num_jobs; j++)
                    for (int o = 0; o < num_machines; o++)
                        if (j * num_machines + o == op)
                            ptime = proc_time[j][o];
                r[i] = r_time[op];
                p[i] = ptime;
                q[i] = q_time[op];
            }
            schrage(ops_on_machine[m], n, r, p, q, seq, heap);
            fix_machine(m, seq, n);
            compute_release();
            compute_tails();
        }
    }
}

/* ── Extract schedule & I/O ─────────────────────────────────── */
static int start_time[MAX_JOBS][MAX_MACHINES];

static void extract_schedule(void)
{
    compute_release();
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
            start_time[j][o] = r_time[j * num_machines + o];
}

static void load_input(const char *file_path)
{
    FILE *input_file = fopen(file_path, "r");
    if (!input_file)
    {
        perror(file_path);
        exit(1);
    }
    fscanf(input_file, "%d %d", &num_jobs, &num_machines);
    for (int job = 0; job < num_jobs; job++)
        for (int op = 0; op < num_machines; op++)
            fscanf(input_file, "%d %d", &machine_id[job][op], &proc_time[job][op]);
    fclose(input_file);
}

static void write_output(const char *file_path, int makespan)
{
    FILE *output_file = fopen(file_path, "w");
    if (!output_file)
    {
        perror(file_path);
        exit(1);
    }
    fprintf(output_file, "%d\n", makespan);
    for (int job = 0; job < num_jobs; job++)
    {
        for (int op = 0; op < num_machines; op++)
        {
            if (op)
                fprintf(output_file, " ");
            fprintf(output_file, "%d", start_time[job][op]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);
}

static void build_graph(void)
{
    total_ops = num_jobs * num_machines;
    memset(adj_cnt, 0, sizeof(adj_cnt));
    memset(pred_cnt, 0, sizeof(pred_cnt));
    memset(machine_fixed, 0, sizeof(machine_fixed));
    memset(ops_on_machine_cnt, 0, sizeof(ops_on_machine_cnt));
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
        {
            int m = machine_id[j][o], op = j * num_machines + o;
            ops_on_machine[m][ops_on_machine_cnt[m]++] = op;
        }
    for (int j = 0; j < num_jobs; j++)
    {
        add_arc(SRC, j * num_machines, 0);
        for (int o = 0; o < num_machines - 1; o++)
            add_arc(j * num_machines + o, j * num_machines + o + 1, proc_time[j][o]);
        add_arc(j * num_machines + num_machines - 1, SNK, proc_time[j][num_machines - 1]);
    }
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <input> <output> <num_threads>\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[3]);
    if (num_threads < 1)
        num_threads = 1;
    omp_set_num_threads(num_threads);

    load_input(argv[1]);
    build_graph();

    struct timespec t0, t1;
    /* timed region: includes thread creation (first omp parallel for) */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    shifting_bottleneck_parallel();
    extract_schedule();
    clock_gettime(CLOCK_MONOTONIC, &t1);

    int makespan = r_time[SNK];
    write_output(argv[2], makespan);

    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("Threads  : %d\n", num_threads);
    printf("Makespan : %d\n", makespan);
    printf("Time     : %.3f ms\n", ms);
    return 0;
}
