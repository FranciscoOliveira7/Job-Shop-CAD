/*
 * Job-Shop Scheduling Problem - Sequential Shifting Bottleneck
 * Computação de Alto Desempenho - MEI IPCA 2025/2026
 *
 * Algorithm: Shifting Bottleneck (Adams, Balas & Zawack, 1988)
 *
 * Overview:
 *   The problem is modelled as a Disjunctive Graph G=(N, A, E) where:
 *     N = all operations + source (s) + sink (t)
 *     A = conjunctive arcs  (job-order precedence, fixed)
 *     E = disjunctive arcs  (same-machine pairs, to be oriented)
 *
 *   The algorithm iterates over unscheduled machines:
 *     1. For every unscheduled machine m, compute release times r_j and
 *        tails q_j for each operation on m (longest paths in current graph).
 *     2. Solve the 1-machine scheduling subproblem 1|r_j|Cmax using
 *        Schrage's algorithm (O(n log n)) — minimises makespan on m.
 *     3. Select the "bottleneck" machine: the one whose 1-machine Cmax
 *        is largest (most critical machine).
 *     4. Fix that machine's sequence: orient its disjunctive arcs in the
 *        graph permanently.
 *     5. Re-optimise all already-fixed machines (one 1-machine solve each)
 *        with the updated release/tail times — this step is key to quality.
 *     6. Repeat until all machines are fixed.
 *
 * 1-machine solver: Schrage's LRT (Largest Remaining Time) algorithm.
 *   Given operations with release times r_j and processing times p_j,
 *   schedules them to minimise Cmax. O(n log n) using a priority queue
 *   (implemented as a max-heap on tail q_j, no internal pointers).
 *
 * Data structures: flat arrays only — no internal pointers.
 *
 * Usage: jobshop_seq <input_file> <output_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>

#define MAX_JOBS 30
#define MAX_MACHINES 30
#define MAX_OPS (MAX_JOBS * MAX_MACHINES) /* total operations */
#define INF 0x3fffffff

/* ── Input ──────────────────────────────────────────────────── */
static int num_jobs;
static int num_machines;
/* machine_id[j][o], proc_time[j][o] */
static int machine_id[MAX_JOBS][MAX_MACHINES];
static int proc_time[MAX_JOBS][MAX_MACHINES];

/*
 * Operation indexing: op_id(j,o) = j*num_machines + o
 * Total operations = num_jobs * num_machines
 * Source = total_ops, Sink = total_ops + 1
 */
static int total_ops;
#define SRC (total_ops)
#define SNK (total_ops + 1)
#define N (total_ops + 2) /* total nodes including src/snk */

/* ── Disjunctive graph (adjacency as flat arrays) ───────────── */
/*
 * Conjunctive arcs (fixed, job order):
 *   src → first op of each job
 *   each op → next op in same job
 *   last op of each job → snk
 *
 * Disjunctive arcs (oriented when machine is fixed):
 *   for each pair of ops on same machine, one direction chosen.
 *
 * We store: for each node u, its successors and the arc weight (= proc_time[u]).
 * Using CSR-like flat arrays.
 */

/* adj[u][0..adj_cnt[u]-1] = successors of u */
/* adj_w[u][k] = weight of arc u→adj[u][k] = proc_time of u */
#define MAX_ADJ 64
static int adj[MAX_OPS + 2][MAX_ADJ];
static int adj_w[MAX_OPS + 2][MAX_ADJ];
static int adj_cnt[MAX_OPS + 2];

/* pred[u][0..pred_cnt[u]-1] = predecessors of u (for tail computation) */
static int pred[MAX_OPS + 2][MAX_ADJ];
static int pred_w[MAX_OPS + 2][MAX_ADJ]; /* weight of arc pred→u = proc_time[pred] */
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
    /* Remove from_node->to_node from adj and to_node's pred list */
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

/* ── Longest-path computation (forward: release times) ─────── */
/*
 * r[u] = length of longest path from SRC to u (= release time of u).
 * Computed by topological-order relaxation (Bellman-Ford on DAG).
 * We use a simple iterative approach: since the graph is a DAG after
 * orienting disjunctive arcs, we iterate until no update occurs.
 * For correctness on the DAG structure this converges in O(N) passes.
 */
static int r_time[MAX_OPS + 2]; /* release times  */
static int q_time[MAX_OPS + 2]; /* tail times (longest path to SNK) */

static void compute_release(void)
{
    for (int node = 0; node < N; node++)
        r_time[node] = 0;
    r_time[SRC] = 0;
    /* Bellman-Ford style: N-1 relaxations */
    for (int pass = 0; pass < N; pass++)
    {
        int changed = 0;
        for (int from_node = 0; from_node < N; from_node++)
        {
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
        }
        if (!changed)
            break;
    }
}

static void compute_tails(void)
{
    /* q[u] = longest path from u to SNK (not including proc_time[u]) */
    for (int node = 0; node < N; node++)
        q_time[node] = 0;
    for (int pass = 0; pass < N; pass++)
    {
        int changed = 0;
        for (int to_node = 0; to_node < N; to_node++)
        {
            for (int edge_index = 0; edge_index < pred_cnt[to_node]; edge_index++)
            {
                int from_node = pred[to_node][edge_index];
                int edge_weight = pred_w[to_node][edge_index]; /* = proc_time[from_node] */
                /* q[u] = max(q[u], w + q[v]) */
                if (edge_weight + q_time[to_node] > q_time[from_node])
                {
                    q_time[from_node] = edge_weight + q_time[to_node];
                    changed = 1;
                }
            }
        }
        if (!changed)
            break;
    }
}

/* ── Machine membership ─────────────────────────────────────── */
/* ops_on_machine[m][0..cnt[m]-1] = op indices on machine m */
static int ops_on_machine[MAX_MACHINES][MAX_JOBS];
static int ops_on_machine_cnt[MAX_MACHINES];

/* ── 1-machine subproblem: Schrage's LRT algorithm ─────────── */
/*
 * Given n operations with release times r[], processing times p[],
 * and tail times q[] (time from end of op to end of schedule),
 * produce a sequence seq[] that minimises Cmax = max(C_j + q_j).
 *
 * Schrage's preemptive LRT (Lageweg, Lenstra, Rinnooy Kan 1978):
 *   - At each step, among available jobs (r_j <= t), schedule the one
 *     with largest q_j (tail). Preempt if a job with larger q arrives.
 *   - For 1|r_j|Cmax this is optimal.
 *
 * We implement the non-preemptive version (which gives a good heuristic
 * for 1|r_j|Cmax, sufficient for Shifting Bottleneck quality goals).
 *
 * Heap: max-heap on q[], flat array, no internal pointers.
 */

#define HEAP_MAX MAX_JOBS

typedef struct
{
    int op;
    int q;
    int r;
    int p;
} HeapItem;
static HeapItem heap[HEAP_MAX];
static int heap_size = 0;

static void heap_push(HeapItem item)
{
    int insert_index = heap_size++;
    heap[insert_index] = item;
    while (insert_index > 0)
    {
        int parent_index = (insert_index - 1) / 2;
        if (heap[parent_index].q < heap[insert_index].q)
        {
            HeapItem tmp = heap[parent_index];
            heap[parent_index] = heap[insert_index];
            heap[insert_index] = tmp;
            insert_index = parent_index;
        }
        else
            break;
    }
}

static HeapItem heap_pop(void)
{
    HeapItem top_item = heap[0];
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
        HeapItem tmp = heap[current_index];
        heap[current_index] = heap[best_index];
        heap[best_index] = tmp;
        current_index = best_index;
    }
    return top_item;
}

/*
 * solve_one_machine: schedule operations ops[0..n-1] on one machine.
 *   r[], p[], q[] are release/processing/tail for each op in ops[].
 *   Output: seq[0..n-1] = scheduling order (indices into ops[]).
 *   Returns: Cmax of the resulting schedule.
 */
static int solve_one_machine(const int *ops, int operation_count,
                             const int *release_times, const int *processing_times,
                             const int *tail_times, int *sequence)
{
    /* unscheduled flags */
    static int done[MAX_JOBS];
    for (int i = 0; i < operation_count; i++)
        done[i] = 0;

    heap_size = 0;
    int current_time = 0;
    /* find min release time to start */
    int min_release = INF;
    for (int i = 0; i < operation_count; i++)
    {
        if (release_times[i] < min_release)
            min_release = release_times[i];
    }
    current_time = min_release;

    int sequence_index = 0;
    int cmax = 0;

    while (sequence_index < operation_count)
    {
        /* push all operations released by current_time */
        for (int i = 0; i < operation_count; i++)
        {
            if (!done[i] && release_times[i] <= current_time)
            {
                HeapItem it;
                it.op = i;
                it.q = tail_times[i];
                it.r = release_times[i];
                it.p = processing_times[i];
                heap_push(it);
                done[i] = 2; /* in heap */
            }
        }
        if (heap_size == 0)
        {
            /* no operation ready: jump to next release time */
            int next_release = INF;
            for (int i = 0; i < operation_count; i++)
            {
                if (!done[i] && release_times[i] < next_release)
                    next_release = release_times[i];
            }
            current_time = next_release;
            continue;
        }
        HeapItem current_op = heap_pop();
        sequence[sequence_index++] = current_op.op;
        current_time += current_op.p;
        int completion_with_tail = current_time + current_op.q;
        if (completion_with_tail > cmax)
            cmax = completion_with_tail;
    }
    return cmax;
}

/* ── Machine fix/unfix state ────────────────────────────────── */
static int machine_fixed[MAX_MACHINES];
/* For each fixed machine, store its sequence */
static int machine_seq[MAX_MACHINES][MAX_JOBS];

/*
 * fix_machine: orient disjunctive arcs for machine m in order seq[].
 * This means: for consecutive ops seq[k] and seq[k+1],
 * add arc seq[k] → seq[k+1] with weight proc_time[seq[k]].
 */
static void fix_machine(int machine, const int *sequence, int operation_count)
{
    /* Remove any existing disjunctive arcs for this machine
       (needed during re-optimisation) */
    for (int i = 0; i < operation_count - 1; i++)
    {
        int from_node = machine_seq[machine][i];
        int to_node = machine_seq[machine][i + 1];
        remove_arc(from_node, to_node);
    }
    /* Add new arcs according to sequence */
    for (int order_index = 0; order_index < operation_count - 1; order_index++)
    {
        int from_node = ops_on_machine[machine][sequence[order_index]];
        int to_node = ops_on_machine[machine][sequence[order_index + 1]];
        int processing_time = 0;
        for (int job = 0; job < num_jobs; job++)
        {
            for (int op = 0; op < num_machines; op++)
            {
                if (job * num_machines + op == from_node)
                    processing_time = proc_time[job][op];
            }
        }
        add_arc(from_node, to_node, processing_time);
        machine_seq[machine][order_index] = from_node;
        machine_seq[machine][order_index + 1] = to_node;
    }
    machine_fixed[machine] = 1;
}

/* ── Compute r and q for ops on machine m ───────────────────── */
static void get_rq_for_machine(int machine, int *release_out,
                               int *processing_out, int *tail_out)
{
    int operation_count = ops_on_machine_cnt[machine];
    for (int i = 0; i < operation_count; i++)
    {
        int operation_id = ops_on_machine[machine][i];
        int processing_time = 0;
        for (int job = 0; job < num_jobs; job++)
        {
            for (int op = 0; op < num_machines; op++)
            {
                if (job * num_machines + op == operation_id)
                    processing_time = proc_time[job][op];
            }
        }
        release_out[i] = r_time[operation_id];
        processing_out[i] = processing_time;
        tail_out[i] = q_time[operation_id];
    }
}

/* ── Shifting Bottleneck main loop ──────────────────────────── */
static void shifting_bottleneck(void)
{
    static int release_times[MAX_JOBS];
    static int processing_times[MAX_JOBS];
    static int tail_times[MAX_JOBS];
    static int sequence[MAX_JOBS];

    int remaining_machines = num_machines;

    while (remaining_machines > 0)
    {
        /* Recompute r and q with current graph */
        compute_release();
        compute_tails();

        /* Find bottleneck: machine with max Cmax among unscheduled */
        int bottleneck_machine = -1;
        int best_cmax = -1;
        int best_sequence[MAX_JOBS];

        for (int machine = 0; machine < num_machines; machine++)
        {
            if (machine_fixed[machine])
                continue;
            int operation_count = ops_on_machine_cnt[machine];
            get_rq_for_machine(machine, release_times, processing_times, tail_times);
            int machine_cmax = solve_one_machine(ops_on_machine[machine], operation_count,
                                                 release_times, processing_times,
                                                 tail_times, sequence);
            if (machine_cmax > best_cmax)
            {
                best_cmax = machine_cmax;
                bottleneck_machine = machine;
                memcpy(best_sequence, sequence, operation_count * sizeof(int));
            }
        }

        /* Fix the bottleneck machine */
        fix_machine(bottleneck_machine, best_sequence,
                    ops_on_machine_cnt[bottleneck_machine]);
        remaining_machines--;

        /* Re-optimise already-fixed machines with updated graph */
        compute_release();
        compute_tails();
        for (int machine = 0; machine < num_machines; machine++)
        {
            if (!machine_fixed[machine] || machine == bottleneck_machine)
                continue;
            int operation_count = ops_on_machine_cnt[machine];
            get_rq_for_machine(machine, release_times, processing_times, tail_times);
            solve_one_machine(ops_on_machine[machine], operation_count,
                              release_times, processing_times,
                              tail_times, sequence);
            fix_machine(machine, sequence, operation_count);
            /* recompute r/q after each re-fix */
            compute_release();
            compute_tails();
        }
    }
}

/* ── Extract start times from release times ─────────────────── */
static int start_time[MAX_JOBS][MAX_MACHINES];

static void extract_schedule(void)
{
    compute_release();
    for (int job = 0; job < num_jobs; job++)
    {
        for (int op = 0; op < num_machines; op++)
        {
            start_time[job][op] = r_time[job * num_machines + op];
        }
    }
}

/* ── File I/O ────────────────────────────────────────────────── */
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
    {
        for (int op = 0; op < num_machines; op++)
        {
            fscanf(input_file, "%d %d", &machine_id[job][op], &proc_time[job][op]);
        }
    }
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

/* ── Build initial conjunctive graph ────────────────────────── */
static void build_graph(void)
{
    total_ops = num_jobs * num_machines;
    memset(adj_cnt, 0, sizeof(adj_cnt));
    memset(pred_cnt, 0, sizeof(pred_cnt));
    memset(machine_fixed, 0, sizeof(machine_fixed));
    memset(ops_on_machine_cnt, 0, sizeof(ops_on_machine_cnt));

    /* Build machine membership */
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
        {
            int m = machine_id[j][o];
            int op = j * num_machines + o;
            ops_on_machine[m][ops_on_machine_cnt[m]++] = op;
        }

    /* Conjunctive arcs: SRC → first op, op[o] → op[o+1], last → SNK */
    for (int j = 0; j < num_jobs; j++)
    {
        add_arc(SRC, j * num_machines, 0);
        for (int o = 0; o < num_machines - 1; o++)
            add_arc(j * num_machines + o, j * num_machines + o + 1, proc_time[j][o]);
        add_arc(j * num_machines + num_machines - 1, SNK,
                proc_time[j][num_machines - 1]);
    }
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <input> <output>\n", argv[0]);
        return 1;
    }

    load_input(argv[1]);
    build_graph();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    shifting_bottleneck();
    extract_schedule();

    clock_gettime(CLOCK_MONOTONIC, &t1);

    int makespan = r_time[SNK];
    write_output(argv[2], makespan);

    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("Makespan : %d\n", makespan);
    printf("Time     : %.3f ms\n", ms);
    return 0;
}
