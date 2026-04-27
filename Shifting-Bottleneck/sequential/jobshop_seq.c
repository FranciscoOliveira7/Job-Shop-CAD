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

#define MAX_JOBS     30
#define MAX_MACHINES 30
#define MAX_OPS      (MAX_JOBS * MAX_MACHINES)   /* total operations */
#define INF          0x3fffffff

/* ── Input ──────────────────────────────────────────────────── */
static int num_jobs;
static int num_machines;
/* machine_id[j][o], proc_time[j][o] */
static int machine_id[MAX_JOBS][MAX_MACHINES];
static int proc_time [MAX_JOBS][MAX_MACHINES];

/*
 * Operation indexing: op_id(j,o) = j*num_machines + o
 * Total operations = num_jobs * num_machines
 * Source = total_ops, Sink = total_ops + 1
 */
static int total_ops;
#define SRC (total_ops)
#define SNK (total_ops + 1)
#define N   (total_ops + 2)   /* total nodes including src/snk */

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
static int adj    [MAX_OPS+2][MAX_ADJ];
static int adj_w  [MAX_OPS+2][MAX_ADJ];
static int adj_cnt[MAX_OPS+2];

/* pred[u][0..pred_cnt[u]-1] = predecessors of u (for tail computation) */
static int pred    [MAX_OPS+2][MAX_ADJ];
static int pred_w  [MAX_OPS+2][MAX_ADJ];   /* weight of arc pred→u = proc_time[pred] */
static int pred_cnt[MAX_OPS+2];

static void add_arc(int u, int v, int w) {
    adj [u][adj_cnt[u]]  = v;
    adj_w[u][adj_cnt[u]] = w;
    adj_cnt[u]++;
    pred [v][pred_cnt[v]]  = u;
    pred_w[v][pred_cnt[v]] = w;
    pred_cnt[v]++;
}

static void remove_arc(int u, int v) {
    /* Remove u→v from adj and v's pred list */
    for (int k = 0; k < adj_cnt[u]; k++) {
        if (adj[u][k] == v) {
            adj[u][k]   = adj[u][adj_cnt[u]-1];
            adj_w[u][k] = adj_w[u][adj_cnt[u]-1];
            adj_cnt[u]--;
            break;
        }
    }
    for (int k = 0; k < pred_cnt[v]; k++) {
        if (pred[v][k] == u) {
            pred[v][k]   = pred[v][pred_cnt[v]-1];
            pred_w[v][k] = pred_w[v][pred_cnt[v]-1];
            pred_cnt[v]--;
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
static int r_time[MAX_OPS+2];   /* release times  */
static int q_time[MAX_OPS+2];   /* tail times (longest path to SNK) */

static void compute_release(void) {
    for (int i = 0; i < N; i++) r_time[i] = 0;
    r_time[SRC] = 0;
    /* Bellman-Ford style: N-1 relaxations */
    for (int pass = 0; pass < N; pass++) {
        int changed = 0;
        for (int u = 0; u < N; u++) {
            for (int k = 0; k < adj_cnt[u]; k++) {
                int v = adj[u][k];
                int w = adj_w[u][k];
                if (r_time[u] + w > r_time[v]) {
                    r_time[v] = r_time[u] + w;
                    changed = 1;
                }
            }
        }
        if (!changed) break;
    }
}

static void compute_tails(void) {
    /* q[u] = longest path from u to SNK (not including proc_time[u]) */
    for (int i = 0; i < N; i++) q_time[i] = 0;
    for (int pass = 0; pass < N; pass++) {
        int changed = 0;
        for (int v = 0; v < N; v++) {
            for (int k = 0; k < pred_cnt[v]; k++) {
                int u = pred[v][k];
                int w = pred_w[v][k];   /* = proc_time[u] */
                /* q[u] = max(q[u], w + q[v]) */
                if (w + q_time[v] > q_time[u]) {
                    q_time[u] = w + q_time[v];
                    changed = 1;
                }
            }
        }
        if (!changed) break;
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

typedef struct { int op; int q; int r; int p; } HeapItem;
static HeapItem heap[HEAP_MAX];
static int heap_size = 0;

static void heap_push(HeapItem item) {
    int i = heap_size++;
    heap[i] = item;
    while (i > 0) {
        int par = (i-1)/2;
        if (heap[par].q < heap[i].q) {
            HeapItem tmp = heap[par]; heap[par] = heap[i]; heap[i] = tmp;
            i = par;
        } else break;
    }
}

static HeapItem heap_pop(void) {
    HeapItem top = heap[0];
    heap[0] = heap[--heap_size];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, best = i;
        if (l < heap_size && heap[l].q > heap[best].q) best = l;
        if (r < heap_size && heap[r].q > heap[best].q) best = r;
        if (best == i) break;
        HeapItem tmp = heap[i]; heap[i] = heap[best]; heap[best] = tmp;
        i = best;
    }
    return top;
}

/*
 * solve_one_machine: schedule operations ops[0..n-1] on one machine.
 *   r[], p[], q[] are release/processing/tail for each op in ops[].
 *   Output: seq[0..n-1] = scheduling order (indices into ops[]).
 *   Returns: Cmax of the resulting schedule.
 */
static int solve_one_machine(const int *ops, int n,
                              const int *r, const int *p, const int *q,
                              int *seq) {
    /* unscheduled flags */
    static int done[MAX_JOBS];
    for (int i = 0; i < n; i++) done[i] = 0;

    heap_size = 0;
    int t = 0;
    /* find min release time to start */
    int min_r = INF;
    for (int i = 0; i < n; i++) if (r[i] < min_r) min_r = r[i];
    t = min_r;

    int seq_idx = 0;
    int cmax = 0;

    while (seq_idx < n) {
        /* push all ops released by t */
        for (int i = 0; i < n; i++) {
            if (!done[i] && r[i] <= t) {
                HeapItem it; it.op=i; it.q=q[i]; it.r=r[i]; it.p=p[i];
                heap_push(it);
                done[i] = 2; /* in heap */
            }
        }
        if (heap_size == 0) {
            /* no op ready: jump to next release time */
            int next = INF;
            for (int i = 0; i < n; i++)
                if (!done[i] && r[i] < next) next = r[i];
            t = next;
            continue;
        }
        HeapItem cur = heap_pop();
        seq[seq_idx++] = cur.op;
        t += cur.p;
        int c = t + cur.q;
        if (c > cmax) cmax = c;
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
static void fix_machine(int m, const int *seq, int n) {
    /* Remove any existing disjunctive arcs for this machine
       (needed during re-optimisation) */
    for (int i = 0; i < n-1; i++) {
        int u = machine_seq[m][i];
        int v = machine_seq[m][i+1];
        remove_arc(u, v);
    }
    /* Add new arcs according to seq */
    for (int k = 0; k < n-1; k++) {
        int u = ops_on_machine[m][seq[k]];
        int v = ops_on_machine[m][seq[k+1]];
        int w = 0;
        /* find proc_time of u */
        for (int j = 0; j < num_jobs; j++)
            for (int o = 0; o < num_machines; o++)
                if (j*num_machines+o == u) w = proc_time[j][o];
        add_arc(u, v, w);
        machine_seq[m][k]   = u;
        machine_seq[m][k+1] = v;
    }
    machine_fixed[m] = 1;
}

/* ── Compute r and q for ops on machine m ───────────────────── */
static void get_rq_for_machine(int m, int *r_out, int *p_out, int *q_out) {
    int n = ops_on_machine_cnt[m];
    for (int i = 0; i < n; i++) {
        int op = ops_on_machine[m][i];
        /* find proc_time of op */
        int ptime = 0;
        for (int j = 0; j < num_jobs; j++)
            for (int o = 0; o < num_machines; o++)
                if (j*num_machines+o == op) ptime = proc_time[j][o];
        r_out[i] = r_time[op];
        p_out[i] = ptime;
        q_out[i] = q_time[op];
    }
}

/* ── Shifting Bottleneck main loop ──────────────────────────── */
static void shifting_bottleneck(void) {
    static int r[MAX_JOBS], p[MAX_JOBS], q[MAX_JOBS], seq[MAX_JOBS];

    int remaining = num_machines;

    while (remaining > 0) {
        /* Recompute r and q with current graph */
        compute_release();
        compute_tails();

        /* Find bottleneck: machine with max Cmax among unscheduled */
        int bottleneck = -1;
        int best_cmax  = -1;
        int best_seq[MAX_JOBS];

        for (int m = 0; m < num_machines; m++) {
            if (machine_fixed[m]) continue;
            int n = ops_on_machine_cnt[m];
            get_rq_for_machine(m, r, p, q);
            int cm = solve_one_machine(ops_on_machine[m], n, r, p, q, seq);
            if (cm > best_cmax) {
                best_cmax = cm;
                bottleneck = m;
                memcpy(best_seq, seq, n * sizeof(int));
            }
        }

        /* Fix the bottleneck machine */
        fix_machine(bottleneck, best_seq, ops_on_machine_cnt[bottleneck]);
        remaining--;

        /* Re-optimise already-fixed machines with updated graph */
        compute_release();
        compute_tails();
        for (int m = 0; m < num_machines; m++) {
            if (!machine_fixed[m] || m == bottleneck) continue;
            int n = ops_on_machine_cnt[m];
            get_rq_for_machine(m, r, p, q);
            solve_one_machine(ops_on_machine[m], n, r, p, q, seq);
            fix_machine(m, seq, n);
            /* recompute r/q after each re-fix */
            compute_release();
            compute_tails();
        }
    }
}

/* ── Extract start times from release times ─────────────────── */
static int start_time[MAX_JOBS][MAX_MACHINES];

static void extract_schedule(void) {
    compute_release();
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
            start_time[j][o] = r_time[j*num_machines+o];
}

/* ── File I/O ────────────────────────────────────────────────── */
static void load_input(const char *fn) {
    FILE *fp = fopen(fn, "r");
    if (!fp) { perror(fn); exit(1); }
    fscanf(fp, "%d %d", &num_jobs, &num_machines);
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
            fscanf(fp, "%d %d", &machine_id[j][o], &proc_time[j][o]);
    fclose(fp);
}

static void write_output(const char *fn, int makespan) {
    FILE *fp = fopen(fn, "w");
    if (!fp) { perror(fn); exit(1); }
    fprintf(fp, "%d\n", makespan);
    for (int j = 0; j < num_jobs; j++) {
        for (int o = 0; o < num_machines; o++) {
            if (o) fprintf(fp, " ");
            fprintf(fp, "%d", start_time[j][o]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* ── Build initial conjunctive graph ────────────────────────── */
static void build_graph(void) {
    total_ops = num_jobs * num_machines;
    memset(adj_cnt,  0, sizeof(adj_cnt));
    memset(pred_cnt, 0, sizeof(pred_cnt));
    memset(machine_fixed, 0, sizeof(machine_fixed));
    memset(ops_on_machine_cnt, 0, sizeof(ops_on_machine_cnt));

    /* Build machine membership */
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++) {
            int m  = machine_id[j][o];
            int op = j*num_machines+o;
            ops_on_machine[m][ops_on_machine_cnt[m]++] = op;
        }

    /* Conjunctive arcs: SRC → first op, op[o] → op[o+1], last → SNK */
    for (int j = 0; j < num_jobs; j++) {
        add_arc(SRC, j*num_machines, 0);
        for (int o = 0; o < num_machines-1; o++)
            add_arc(j*num_machines+o, j*num_machines+o+1, proc_time[j][o]);
        add_arc(j*num_machines+num_machines-1, SNK,
                proc_time[j][num_machines-1]);
    }
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 3) {
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

    double ms = (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("Makespan : %d\n", makespan);
    printf("Time     : %.3f ms\n", ms);
    return 0;
}
