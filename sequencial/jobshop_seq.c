/*
 * Job-Shop Scheduling Problem - Sequential Branch & Bound
 * Computação de Alto Desempenho - MEI IPCA 2025/2026
 *
 * Algorithm: Branch & Bound with two lower bounds:
 *   LB1 (job bound)     – earliest completion of remaining ops per job.
 *   LB2 (machine bound) – remaining workload per machine + machine free time.
 *   lower_bound = max(LB1, LB2)
 *
 * Warm-start: greedy EST provides an initial upper bound for aggressive pruning.
 * Time-limit: if TIME_LIMIT_SEC seconds elapse, the search stops and the best
 *             solution found so far is returned (it will always be at least as
 *             good as the greedy solution because greedy_init() runs first).
 *
 * Data structures: flat arrays only — no internal pointer usage in structures.
 *   - stack[] : heap-allocated array used as a DFS LIFO stack.
 *   - best_start[][] : flat 2-D array for the best schedule found.
 *
 * Usage: jobshop_seq <input_file> <output_file> [time_limit_sec]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_JOBS 30
#define MAX_MACHINES 30
#define STACK_CAPACITY 2000000
#define DEFAULT_LIMIT 60 /* seconds */

/* ── global time limit ──────────────────────────────────────── */
static double time_limit_sec = DEFAULT_LIMIT;
static struct timespec t_start_global;

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
/* remaining_time[j][o] = sum of proc_time[j][o .. num_machines-1] */
static int remaining_time[MAX_JOBS][MAX_MACHINES + 1];

/* ── node ───────────────────────────────────────────────────── */
typedef struct
{
    short start[MAX_JOBS][MAX_MACHINES]; /* -1 = unscheduled  */
    short job_ready[MAX_JOBS];           /* earliest next-op  */
    short mach_free[MAX_MACHINES];       /* earliest mach free*/
    unsigned char next_op[MAX_JOBS];     /* next op index     */
    short scheduled;                     /* ops done so far   */
    short lower_bound;
} Node;

/* ── DFS stack (flat array on heap, no internal pointers) ───── */
static Node *stack = NULL;
static int stack_top = -1;

static inline void push(const Node *n)
{
    if (stack_top + 1 >= STACK_CAPACITY)
    {
        fprintf(stderr, "Stack overflow – increase STACK_CAPACITY\n");
        exit(EXIT_FAILURE);
    }
    stack[++stack_top] = *n;
}
static inline int pop(Node *n)
{
    if (stack_top < 0)
        return 0;
    *n = stack[stack_top--];
    return 1;
}

/* ── best solution ──────────────────────────────────────────── */
static int best_makespan;
static short best_start[MAX_JOBS][MAX_MACHINES];

/* ────────────────────────────────────────────────────────────
 * Lower bound
 * LB1: for each job j with ops remaining from next_op[j]:
 *        earliest_start + remaining_time[j][next_op[j]]
 * LB2: for each machine m:
 *        mach_free[m] + sum of proc_time of all unscheduled ops on m
 * lb = max over all jobs (LB1) and all machines (LB2)
 * ──────────────────────────────────────────────────────────── */
static int lower_bound(const Node *n)
{
    int lb = 0;

    /* LB1 – job-based */
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

    /* LB2 – machine-based */
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

/* ── greedy warm-start (Earliest Start Time per job, in order) ─ */
static void greedy_init(void)
{
    int jr[MAX_JOBS] = {0};
    int mf[MAX_MACHINES] = {0};
    best_makespan = 0;

    for (int j = 0; j < num_jobs; j++)
    {
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
}

/* ── branch & bound (iterative DFS) ────────────────────────── */
static void branch_and_bound(void)
{
    /* Initialise root node */
    Node root;
    memset(&root, 0, sizeof(root));
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
            root.start[j][o] = -1;
    push(&root);

    int total_ops = num_jobs * num_machines;
    int check_every = 50000; /* check clock every N iterations */
    int iter = 0;
    Node cur;

    while (pop(&cur))
    {
        /* Periodic time-limit check */
        if (++iter % check_every == 0)
        {
            if (elapsed_sec_now() >= time_limit_sec)
                break;
        }

        /* Prune */
        if (cur.lower_bound >= best_makespan)
            continue;

        /* Leaf: update best solution */
        if (cur.scheduled == total_ops)
        {
            int ms = 0;
            for (int j = 0; j < num_jobs; j++)
            {
                int f = cur.start[j][num_machines - 1] + proc_time[j][num_machines - 1];
                if (f > ms)
                    ms = f;
            }
            if (ms < best_makespan)
            {
                best_makespan = ms;
                memcpy(best_start, cur.start, sizeof(cur.start));
            }
            continue;
        }

        /*
         * Branch: for each job with ops remaining, schedule its next op
         * at its earliest possible start time (EST = max(job_ready, mach_free)).
         * Children are pushed in reverse job order so job-0 is explored first
         * (LIFO stack).
         */
        for (int j = num_jobs - 1; j >= 0; j--)
        {
            int op = cur.next_op[j];
            if (op >= num_machines)
                continue;

            Node child = cur; /* copy full state */

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

            if (child.lower_bound < best_makespan)
                push(&child);
        }
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
        fprintf(stderr, "Bad header in %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_machines; o++)
            if (fscanf(fp, "%d %d", &machine_id[j][o], &proc_time[j][o]) != 2)
            {
                fprintf(stderr, "Bad data at j=%d o=%d\n", j, o);
                exit(EXIT_FAILURE);
            }
    fclose(fp);

    /* Precompute suffix sums of processing times per job */
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
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <input_file> <output_file> [time_limit_sec]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (argc >= 4)
        time_limit_sec = atof(argv[3]);

    load_input(argv[1]);

    stack = (Node *)malloc((size_t)STACK_CAPACITY * sizeof(Node));
    if (!stack)
    {
        perror("malloc");
        return EXIT_FAILURE;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_start_global);

    greedy_init();      /* always gives a valid schedule first */
    branch_and_bound(); /* improves it within the time limit   */

    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed_ms = (t_end.tv_sec - t_start_global.tv_sec) * 1000.0 + (t_end.tv_nsec - t_start_global.tv_nsec) / 1e6;

    write_output(argv[2]);
    free(stack);

    printf("Makespan : %d\n", best_makespan);
    printf("Time     : %.3f ms\n", elapsed_ms);
    return EXIT_SUCCESS;
}