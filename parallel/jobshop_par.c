/*
 * Job-Shop Scheduling Problem - Parallel Branch & Bound (OpenMP)
 * Computação de Alto Desempenho - MEI IPCA 2025/2026
 *
 * ─── Foster Methodology ─────────────────────────────────────────────────────
 * PARTITION   : each node of the B&B search tree is an independent task
 *               (its lower-bound computation and child generation depend only
 *               on the node's own state).
 * COMMUNICATION: threads share a global work stack and a global best solution.
 *               Both are protected by omp_lock_t mutexes.
 * AGGLOMERATION: each thread dequeues and fully processes one node per step
 *               (compute LB, expand children, push valid children).
 * MAPPING      : OpenMP creates a fixed-size thread pool; the number of
 *               threads is supplied as a command-line argument.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 * Shared read-only globals:
 *   num_jobs, num_machines, machine_id[][], proc_time[][], remaining_time[][]
 *
 * Shared read-write globals (critical sections noted):
 *   stack[], stack_top       → CS1, protected by stack_lock
 *   best_makespan, best_start → CS2, protected by best_lock
 *   active_threads            → CS3, protected by active_lock
 *
 * Thread-local variables:
 *   cur, child  (Node structs declared inside worker(), on each thread's stack)
 *
 * Mutual exclusion technique: omp_lock_t (OpenMP spin-lock).
 *   - stack_lock : guards all stack push/pop operations.
 *   - best_lock  : guards reads/writes of best_makespan and best_start.
 *   - active_lock: guards the active_threads counter used for termination.
 *
 * Race conditions prevented:
 *   - Two threads could simultaneously read best_makespan < child.lb and both
 *     push the same child → best_lock acquired before every LB comparison.
 *   - Two threads could simultaneously update best_makespan → best_lock
 *     ensures only one update at a time.
 *   - Termination detection: a thread must NOT exit while another thread
 *     holds a node (which may produce new work) → active_lock + active_threads.
 *
 * Usage: jobshop_par <input_file> <output_file> <num_threads> [time_limit_sec]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define MAX_JOBS 30
#define MAX_MACHINES 30
#define STACK_CAPACITY 2000000
#define DEFAULT_LIMIT 60 /* seconds */

/* ── global time limit ──────────────────────────────────────── */
static double time_limit_sec = DEFAULT_LIMIT;
static struct timespec t_start_global;
static volatile int timed_out = 0; /* written once, read by all threads */

static inline double elapsed_sec_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - t_start_global.tv_sec) + (now.tv_nsec - t_start_global.tv_nsec) / 1e9;
}

/* ── input data (read-only after loading) ──────────────────── */
static int num_jobs;
static int num_machines;
static int machine_id[MAX_JOBS][MAX_MACHINES];
static int proc_time[MAX_JOBS][MAX_MACHINES];
static int remaining_time[MAX_JOBS][MAX_MACHINES + 1];

/* ── node ───────────────────────────────────────────────────── */
typedef struct
{
    short start[MAX_JOBS][MAX_MACHINES];
    short job_ready[MAX_JOBS];
    short mach_free[MAX_MACHINES];
    unsigned char next_op[MAX_JOBS];
    short scheduled;
    short lower_bound;
} Node;

/* ── shared work stack (CS1) ────────────────────────────────── */
static Node *stack = NULL;
static int stack_top = -1;
static omp_lock_t stack_lock;

static void stack_push(const Node *n)
{
    omp_set_lock(&stack_lock); /* CS1 enter */
    if (stack_top + 1 < STACK_CAPACITY)
        stack[++stack_top] = *n;
    else
        fprintf(stderr, "Warning: stack full, node discarded\n");
    omp_unset_lock(&stack_lock); /* CS1 exit  */
}

static int stack_pop(Node *n)
{
    int got = 0;
    omp_set_lock(&stack_lock); /* CS1 enter */
    if (stack_top >= 0)
    {
        *n = stack[stack_top--];
        got = 1;
    }
    omp_unset_lock(&stack_lock); /* CS1 exit  */
    return got;
}

static int stack_size_locked(void)
{
    omp_set_lock(&stack_lock);
    int s = stack_top + 1;
    omp_unset_lock(&stack_lock);
    return s;
}

/* ── shared best solution (CS2) ─────────────────────────────── */
static int best_makespan;
static short best_start[MAX_JOBS][MAX_MACHINES];
static omp_lock_t best_lock;

/* ── active thread counter (CS3) ────────────────────────────── */
static int active_threads = 0;
static omp_lock_t active_lock;

/* ── lower bound ────────────────────────────────────────────── */
static int lower_bound(const Node *n)
{
    int lb = 0;

    /* LB1: job-based */
    for (int j = 0; j < num_jobs; j++)
    {
        int op = n->next_op[j];
        if (op >= num_machines)
            continue;
        int earliest = n->job_ready[j];
        if (n->mach_free[machine_id[j][op]] > earliest)
            earliest = n->mach_free[machine_id[j][op]];
        int f = earliest + remaining_time[j][op];
        if (f > lb)
            lb = f;
    }

    /* LB2: machine-based */
    int mrem[MAX_MACHINES] = {0};
    for (int j = 0; j < num_jobs; j++)
        for (int o = (int)n->next_op[j]; o < num_machines; o++)
            mrem[machine_id[j][o]] += proc_time[j][o];
    for (int m = 0; m < num_machines; m++)
    {
        int f = (int)n->mach_free[m] + mrem[m];
        if (f > lb)
            lb = f;
    }
    return lb;
}

/* ── greedy warm-start ──────────────────────────────────────── */
static void greedy_init(void)
{
    int jr[MAX_JOBS] = {0};
    int mf[MAX_MACHINES] = {0};
    best_makespan = 0;
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
        {
            int m = machine_id[j][o];
            int ts = jr[j] > mf[m] ? jr[j] : mf[m];
            best_start[j][o] = (short)ts;
            int te = ts + proc_time[j][o];
            jr[j] = te;
            mf[m] = te;
            if (te > best_makespan)
                best_makespan = te;
        }
}

/* ── worker function ────────────────────────────────────────── */
static void worker(void)
{
    int total_ops = num_jobs * num_machines;
    int check_every = 10000;
    int iter = 0;

    while (!timed_out)
    {

        /* ─ dequeue one node ─ */
        Node cur;
        if (!stack_pop(&cur))
        {
            /* Stack empty: check termination */
            omp_set_lock(&active_lock);
            int busy = active_threads;
            omp_unset_lock(&active_lock);
            if (busy == 0 && stack_size_locked() == 0)
                break;
            continue; /* spin while other threads may push work */
        }

        /* Mark this thread as doing work (CS3) */
        omp_set_lock(&active_lock);
        active_threads++;
        omp_unset_lock(&active_lock);

        /* Periodic time check */
        if (++iter % check_every == 0)
        {
            if (elapsed_sec_now() >= time_limit_sec)
            {
                timed_out = 1;
                omp_set_lock(&active_lock);
                active_threads--;
                omp_unset_lock(&active_lock);
                break;
            }
        }

        /* Read current best for pruning (CS2) */
        omp_set_lock(&best_lock);
        int cur_best = best_makespan;
        omp_unset_lock(&best_lock);

        /* Prune */
        if (cur.lower_bound >= cur_best)
        {
            omp_set_lock(&active_lock);
            active_threads--;
            omp_unset_lock(&active_lock);
            continue;
        }

        /* Leaf: update best solution (CS2) */
        if (cur.scheduled == total_ops)
        {
            int ms = 0;
            for (int j = 0; j < num_jobs; j++)
            {
                int f = cur.start[j][num_machines - 1] + proc_time[j][num_machines - 1];
                if (f > ms)
                    ms = f;
            }
            omp_set_lock(&best_lock); /* CS2 enter */
            if (ms < best_makespan)
            {
                best_makespan = ms;
                memcpy(best_start, cur.start, sizeof(cur.start));
            }
            omp_unset_lock(&best_lock); /* CS2 exit  */

            omp_set_lock(&active_lock);
            active_threads--;
            omp_unset_lock(&active_lock);
            continue;
        }

        /* Branch: expand children, push in reverse order */
        for (int j = num_jobs - 1; j >= 0; j--)
        {
            int op = cur.next_op[j];
            if (op >= num_machines)
                continue;

            Node child = cur;
            int m = machine_id[j][op];
            int ts = cur.job_ready[j] > cur.mach_free[m]
                         ? cur.job_ready[j]
                         : cur.mach_free[m];
            int te = ts + proc_time[j][op];

            child.start[j][op] = (short)ts;
            child.job_ready[j] = (short)te;
            child.mach_free[m] = (short)te;
            child.next_op[j]++;
            child.scheduled++;
            child.lower_bound = (short)lower_bound(&child);

            omp_set_lock(&best_lock);
            int push_ok = (child.lower_bound < best_makespan);
            omp_unset_lock(&best_lock);

            if (push_ok)
                stack_push(&child); /* CS1 */
        }

        omp_set_lock(&active_lock);
        active_threads--;
        omp_unset_lock(&active_lock);
    }
}

/* ── file I/O ───────────────────────────────────────────────── */
static void load_input(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    if (fscanf(fp, "%d %d", &num_jobs, &num_machines) != 2)
    {
        fprintf(stderr, "Bad header\n");
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
            if (fscanf(fp, "%d %d", &machine_id[j][o], &proc_time[j][o]) != 2)
            {
                fprintf(stderr, "Bad data j=%d o=%d\n", j, o);
                exit(EXIT_FAILURE);
            }
    fclose(fp);
    for (int j = 0; j < num_jobs; j++)
    {
        remaining_time[j][num_machines] = 0;
        for (int o = num_machines - 1; o >= 0; o--)
            remaining_time[j][o] = remaining_time[j][o + 1] + proc_time[j][o];
    }
}

static void write_output(const char *filename)
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "%d\n", best_makespan);
    for (int j = 0; j < num_jobs; j++)
    {
        for (int o = 0; o < num_machines; o++)
        {
            if (o)
                fprintf(fp, " ");
            fprintf(fp, "%d", (int)best_start[j][o]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* ── main ───────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr,
                "Usage: %s <input_file> <output_file> <num_threads> [time_limit_sec]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    int num_threads = atoi(argv[3]);
    if (num_threads < 1)
        num_threads = 1;
    if (argc >= 5)
        time_limit_sec = atof(argv[4]);
    omp_set_num_threads(num_threads);

    load_input(argv[1]);

    stack = (Node *)malloc((size_t)STACK_CAPACITY * sizeof(Node));
    if (!stack)
    {
        perror("malloc");
        return EXIT_FAILURE;
    }

    omp_init_lock(&stack_lock);
    omp_init_lock(&best_lock);
    omp_init_lock(&active_lock);

    /* ── timed region starts here (includes thread create/join) ── */
    clock_gettime(CLOCK_MONOTONIC, &t_start_global);

    greedy_init();

    /* Push root before spawning threads (no lock needed yet) */
    {
        Node root;
        memset(&root, 0, sizeof(root));
        for (int j = 0; j < num_jobs; j++)
            for (int o = 0; o < num_machines; o++)
                root.start[j][o] = -1;
        stack[++stack_top] = root;
    }

#pragma omp parallel
    {
        worker();
    }
    /* ── timed region ends here ── */

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ms = (t_end.tv_sec - t_start_global.tv_sec) * 1000.0 + (t_end.tv_nsec - t_start_global.tv_nsec) / 1e6;

    omp_destroy_lock(&stack_lock);
    omp_destroy_lock(&best_lock);
    omp_destroy_lock(&active_lock);
    free(stack);

    write_output(argv[2]);
    printf("Threads  : %d\n", num_threads);
    printf("Makespan : %d\n", best_makespan);
    printf("Time     : %.3f ms\n", elapsed_ms);
    printf("TimedOut : %s\n", timed_out ? "yes (best-found returned)" : "no (optimal)");
    return EXIT_SUCCESS;
}