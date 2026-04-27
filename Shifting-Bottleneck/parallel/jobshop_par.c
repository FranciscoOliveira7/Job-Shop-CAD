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

#define MAX_JOBS     30
#define MAX_MACHINES 30
#define MAX_OPS      (MAX_JOBS * MAX_MACHINES)
#define MAX_ADJ      64
#define INF          0x3fffffff

/* ── Input ──────────────────────────────────────────────────── */
static int num_jobs, num_machines, total_ops;
static int machine_id[MAX_JOBS][MAX_MACHINES];
static int proc_time [MAX_JOBS][MAX_MACHINES];

#define SRC (total_ops)
#define SNK (total_ops + 1)
#define N   (total_ops + 2)

/* ── Disjunctive graph ──────────────────────────────────────── */
static int adj    [MAX_OPS+2][MAX_ADJ];
static int adj_w  [MAX_OPS+2][MAX_ADJ];
static int adj_cnt[MAX_OPS+2];
static int pred    [MAX_OPS+2][MAX_ADJ];
static int pred_w  [MAX_OPS+2][MAX_ADJ];
static int pred_cnt[MAX_OPS+2];

static void add_arc(int u, int v, int w) {
    adj [u][adj_cnt[u]]  = v; adj_w[u][adj_cnt[u]] = w; adj_cnt[u]++;
    pred[v][pred_cnt[v]] = u; pred_w[v][pred_cnt[v]] = w; pred_cnt[v]++;
}
static void remove_arc(int u, int v) {
    for (int k=0;k<adj_cnt[u];k++) if(adj[u][k]==v){
        adj[u][k]=adj[u][adj_cnt[u]-1]; adj_w[u][k]=adj_w[u][adj_cnt[u]-1];
        adj_cnt[u]--; break; }
    for (int k=0;k<pred_cnt[v];k++) if(pred[v][k]==u){
        pred[v][k]=pred[v][pred_cnt[v]-1]; pred_w[v][k]=pred_w[v][pred_cnt[v]-1];
        pred_cnt[v]--; break; }
}

/* ── Release / tail times (computed sequentially) ───────────── */
/* These arrays are written ONLY in compute_release/compute_tails,
   which are called BEFORE any parallel region — so they are
   read-only from the threads' perspective. */
static int r_time[MAX_OPS+2];
static int q_time[MAX_OPS+2];

static void compute_release(void) {
    for (int i=0;i<N;i++) r_time[i]=0;
    for (int pass=0;pass<N;pass++) {
        int ch=0;
        for (int u=0;u<N;u++)
            for (int k=0;k<adj_cnt[u];k++) {
                int v=adj[u][k], w=adj_w[u][k];
                if (r_time[u]+w>r_time[v]) { r_time[v]=r_time[u]+w; ch=1; }
            }
        if (!ch) break;
    }
}
static void compute_tails(void) {
    for (int i=0;i<N;i++) q_time[i]=0;
    for (int pass=0;pass<N;pass++) {
        int ch=0;
        for (int v=0;v<N;v++)
            for (int k=0;k<pred_cnt[v];k++) {
                int u=pred[v][k], w=pred_w[v][k];
                if (w+q_time[v]>q_time[u]) { q_time[u]=w+q_time[v]; ch=1; }
            }
        if (!ch) break;
    }
}

/* ── Machine membership ─────────────────────────────────────── */
static int ops_on_machine    [MAX_MACHINES][MAX_JOBS];
static int ops_on_machine_cnt[MAX_MACHINES];

/* ── Per-machine Cmax results (written by parallel threads) ─── */
/* Each machine m writes to cmax_result[m] and seq_result[m][].
   Since each thread writes to a DIFFERENT machine's slot,
   there is no race condition — no lock needed. */
static int cmax_result[MAX_MACHINES];
static int seq_result [MAX_MACHINES][MAX_JOBS];

/* ── 1-machine Schrage solver (thread-safe, uses only locals) ── */
#define HEAP_MAX MAX_JOBS
typedef struct { int op,q,r,p; } HItem;

static int schrage(const int *ops, int n,
                   const int *r, const int *p, const int *q,
                   int *seq,
                   HItem *heap) {   /* heap is caller-provided (thread-local) */
    static __thread int done[MAX_JOBS];
    for (int i=0;i<n;i++) done[i]=0;
    int hsz=0;

    int min_r=INF;
    for (int i=0;i<n;i++) if(r[i]<min_r) min_r=r[i];
    int t=min_r, sidx=0, cmax=0;

    while (sidx<n) {
        for (int i=0;i<n;i++) {
            if (!done[i] && r[i]<=t) {
                HItem it={i,q[i],r[i],p[i]};
                int pos=hsz++;  heap[pos]=it;
                while(pos>0){ int par=(pos-1)/2;
                    if(heap[par].q<heap[pos].q){
                        HItem tmp=heap[par]; heap[par]=heap[pos]; heap[pos]=tmp;
                        pos=par;} else break; }
                done[i]=2;
            }
        }
        if (!hsz) {
            int nxt=INF;
            for (int i=0;i<n;i++) if(!done[i]&&r[i]<nxt) nxt=r[i];
            t=nxt; continue;
        }
        HItem cur=heap[0]; heap[0]=heap[--hsz];
        int i=0;
        while(1){ int l=2*i+1,r2=2*i+2,best=i;
            if(l<hsz&&heap[l].q>heap[best].q) best=l;
            if(r2<hsz&&heap[r2].q>heap[best].q) best=r2;
            if(best==i) break;
            HItem tmp=heap[i]; heap[i]=heap[best]; heap[best]=tmp; i=best; }
        seq[sidx++]=cur.op;
        t+=cur.p;
        int c=t+cur.q; if(c>cmax) cmax=c;
    }
    return cmax;
}


/* ── Machine fix state ──────────────────────────────────────── */
static int machine_fixed[MAX_MACHINES];
static int machine_seq  [MAX_MACHINES][MAX_JOBS];

static void fix_machine(int m, const int *seq, int n) {
    for (int i=0;i<n-1;i++) remove_arc(machine_seq[m][i], machine_seq[m][i+1]);
    for (int k=0;k<n-1;k++) {
        int u=ops_on_machine[m][seq[k]], v=ops_on_machine[m][seq[k+1]];
        int w=0;
        for (int j=0;j<num_jobs;j++)
            for (int o=0;o<num_machines;o++)
                if (j*num_machines+o==u) w=proc_time[j][o];
        add_arc(u,v,w);
        machine_seq[m][k]=u; machine_seq[m][k+1]=v;
    }
    machine_fixed[m]=1;
}

/* ── Parallel Shifting Bottleneck ───────────────────────────── */
static void shifting_bottleneck_parallel(void) {
    int remaining = num_machines;

    while (remaining > 0) {

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
        #pragma omp parallel for schedule(static) default(none) \
            shared(machine_fixed, ops_on_machine, \
                   ops_on_machine_cnt, proc_time, machine_id, num_jobs, \
                   num_machines, r_time, q_time, \
                   cmax_result, seq_result)
        for (int m = 0; m < num_machines; m++) {
            if (machine_fixed[m]) {
                cmax_result[m] = -1;
                continue;
            }
            /* Thread-local arrays — stack-allocated, no sharing */
            int r[MAX_JOBS], p[MAX_JOBS], q[MAX_JOBS], seq[MAX_JOBS];
            HItem heap[MAX_JOBS];

            int n = ops_on_machine_cnt[m];
            /* Fill r/p/q from read-only shared arrays */
            for (int i = 0; i < n; i++) {
                int op = ops_on_machine[m][i];
                int ptime = 0;
                for (int j = 0; j < num_jobs; j++)
                    for (int o = 0; o < num_machines; o++)
                        if (j*num_machines+o == op) ptime = proc_time[j][o];
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
        int bottleneck = -1, best_cmax = -1;
        for (int m = 0; m < num_machines; m++)
            if (cmax_result[m] > best_cmax) {
                best_cmax = cmax_result[m];
                bottleneck = m;
            }

        fix_machine(bottleneck, seq_result[bottleneck],
                    ops_on_machine_cnt[bottleneck]);
        remaining--;

        /* ── Re-optimise fixed machines (sequential per fix) ── */
        compute_release();
        compute_tails();
        for (int m = 0; m < num_machines; m++) {
            if (!machine_fixed[m] || m == bottleneck) continue;
            int n = ops_on_machine_cnt[m];
            int r[MAX_JOBS], p[MAX_JOBS], q[MAX_JOBS], seq[MAX_JOBS];
            HItem heap[MAX_JOBS];
            for (int i=0;i<n;i++) {
                int op=ops_on_machine[m][i], ptime=0;
                for (int j=0;j<num_jobs;j++)
                    for (int o=0;o<num_machines;o++)
                        if (j*num_machines+o==op) ptime=proc_time[j][o];
                r[i]=r_time[op]; p[i]=ptime; q[i]=q_time[op];
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

static void extract_schedule(void) {
    compute_release();
    for (int j=0;j<num_jobs;j++)
        for (int o=0;o<num_machines;o++)
            start_time[j][o] = r_time[j*num_machines+o];
}

static void load_input(const char *fn) {
    FILE *fp=fopen(fn,"r");
    if (!fp){perror(fn);exit(1);}
    fscanf(fp,"%d %d",&num_jobs,&num_machines);
    for (int j=0;j<num_jobs;j++)
        for (int o=0;o<num_machines;o++)
            fscanf(fp,"%d %d",&machine_id[j][o],&proc_time[j][o]);
    fclose(fp);
}

static void write_output(const char *fn, int makespan) {
    FILE *fp=fopen(fn,"w");
    if (!fp){perror(fn);exit(1);}
    fprintf(fp,"%d\n",makespan);
    for (int j=0;j<num_jobs;j++){
        for (int o=0;o<num_machines;o++){if(o)fprintf(fp," ");fprintf(fp,"%d",start_time[j][o]);}
        fprintf(fp,"\n");
    }
    fclose(fp);
}

static void build_graph(void) {
    total_ops=num_jobs*num_machines;
    memset(adj_cnt,0,sizeof(adj_cnt)); memset(pred_cnt,0,sizeof(pred_cnt));
    memset(machine_fixed,0,sizeof(machine_fixed));
    memset(ops_on_machine_cnt,0,sizeof(ops_on_machine_cnt));
    for (int j=0;j<num_jobs;j++)
        for (int o=0;o<num_machines;o++){
            int m=machine_id[j][o], op=j*num_machines+o;
            ops_on_machine[m][ops_on_machine_cnt[m]++]=op;
        }
    for (int j=0;j<num_jobs;j++){
        add_arc(SRC, j*num_machines, 0);
        for (int o=0;o<num_machines-1;o++)
            add_arc(j*num_machines+o, j*num_machines+o+1, proc_time[j][o]);
        add_arc(j*num_machines+num_machines-1, SNK, proc_time[j][num_machines-1]);
    }
}

/* ── main ────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr,"Usage: %s <input> <output> <num_threads>\n",argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[3]);
    if (num_threads<1) num_threads=1;
    omp_set_num_threads(num_threads);

    load_input(argv[1]);
    build_graph();

    struct timespec t0,t1;
    /* timed region: includes thread creation (first omp parallel for) */
    clock_gettime(CLOCK_MONOTONIC,&t0);
    shifting_bottleneck_parallel();
    extract_schedule();
    clock_gettime(CLOCK_MONOTONIC,&t1);

    int makespan=r_time[SNK];
    write_output(argv[2],makespan);

    double ms=(t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("Threads  : %d\n",num_threads);
    printf("Makespan : %d\n",makespan);
    printf("Time     : %.3f ms\n",ms);
    return 0;
}
