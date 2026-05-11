# Job-Shop Scheduling Code Explanation

This project solves a **job-shop scheduling problem** using the **shifting bottleneck** method.

If you are looking at the code for the first time, the main idea is:

1. Each job has a fixed order of operations.
2. Each operation must run on a specific machine.
3. The code tries to find start times that make the whole schedule finish as early as possible.
4. It does this by repeatedly choosing the most critical machine, fixing its order, and updating the rest of the schedule.

There are two versions:

- `sequential/jobshop_seq.c` - single-threaded version.
- `parallel/jobshop_par.c` - OpenMP version that evaluates machines in parallel.

---

## 1. The problem being solved

A job-shop instance gives you:

- a number of jobs
- a number of machines
- for each job, a list of operations
- for each operation:
  - which machine it needs
  - how long it takes

The goal is to assign start times so that:

- operations of the same job happen in the given order
- a machine can only process one operation at a time
- the total finishing time, called the **makespan**, is as small as possible

---

## 2. How the code represents the problem

The code turns the schedule into a **graph**.

### Nodes

Each operation becomes a node.

The code also adds:

- `SRC` = source node
- `SNK` = sink node

So the graph contains:

- all operations
- one source
- one sink

### Edges

There are two kinds of edges:

- **conjunctive edges**: fixed job order edges
- **disjunctive edges**: machine conflicts that get oriented later

### Why a graph?

Because the earliest start time of an operation is the length of the longest path from the source to that operation.

That is what `compute_release()` calculates.

The reverse longest-path values are the tail times, calculated by `compute_tails()`.

---

## 3. Important arrays

### Input arrays

- `machine_id[j][o]` - which machine job `j`, operation `o` needs
- `proc_time[j][o]` - how long that operation takes

### Graph arrays

- `adj`, `adj_w`, `adj_cnt` - outgoing edges and their weights
- `pred`, `pred_w`, `pred_cnt` - incoming edges and their weights

### Scheduling arrays

- `r_time[]` - release times, or earliest start times
- `q_time[]` - tail times, or remaining time to the end
- `ops_on_machine[][]` - all operations that need each machine
- `machine_fixed[]` - whether a machine has already been fixed
- `machine_seq[][]` - the chosen order of operations on each fixed machine

### Output arrays

- `start_time[][]` - final start time of each job operation

---

## 4. What the key functions do

### `add_arc()`

Adds an edge to the graph in both forward and reverse form.

### `remove_arc()`

Deletes an edge from the graph.
This is used when a fixed machine gets re-optimized.

### `compute_release()`

Computes the earliest time each node can start.
In simple terms, it asks:

- if everything before this operation is already fixed, how early can this operation begin?

### `compute_tails()`

Computes how much work remains after each node.
In simple terms, it asks:

- if this operation finishes now, how much time is still left until the end?

### `operation_time(op)`

Returns the processing time of one operation.
This is just a helper so the code does not repeat the lookup logic.

### `build_machine_problem()`

Builds the small 1-machine scheduling problem for one machine.
It fills:

- `r[]` from `r_time[]`
- `p[]` from `proc_time[]`
- `q[]` from `q_time[]`

### `schrage()` / `solve_one_machine()`

Solves the smaller problem of ordering operations on a single machine.

This is the heart of the per-machine step.
The rule is basically:

- schedule available operations
- prefer the one with the largest tail `q`

That helps minimize the completion time for that machine.

### `fix_machine()`

Takes the chosen order for one machine and permanently adds the corresponding edges to the graph.

This is the step where the algorithm says:

- “this machine’s order is now fixed, do not change it anymore”

### `extract_schedule()`

After all machines are fixed, this reads the final release times and stores them as the actual start times.

### `load_input()`

Reads the input file.

### `write_output()`

Writes the makespan and the final start-time table.

### `build_graph()`

Creates the initial graph from the input data.
It also fills `ops_on_machine[]` so the code knows which operations compete for each machine.

---

## 5. How the algorithm works

The algorithm repeats this loop until every machine is fixed:

1. Recompute release times and tail times.
2. For every unfixed machine, solve its 1-machine problem.
3. Pick the machine with the worst result.
4. Fix that machine in the graph.
5. Re-optimize the machines that were already fixed.

That is why it is called **shifting bottleneck**:

- bottleneck = the most critical machine
- shifting = after fixing one machine, another machine may become the new bottleneck

---

## 6. Sequential version vs parallel version

### Sequential version

In `sequential/jobshop_seq.c`, the code checks machines one by one.

The flow is straightforward:

- compute `r_time` and `q_time`
- test each machine
- keep the machine with the largest `Cmax`
- fix it
- repeat

### Parallel version

In `parallel/jobshop_par.c`, the same logic is used, but the machine evaluation step runs in parallel with OpenMP.

That means:

- multiple machines are tested at the same time
- each thread works on one machine at a time
- only the evaluation step is parallel
- the fixing step stays sequential because it changes the shared graph

This is the main reason the parallel version is faster.

---

## 7. What happens in the parallel file

The parallel version has this basic structure:

1. Build the graph.
2. Compute release and tail times.
3. Run `#pragma omp parallel for` over all unfixed machines.
4. Each thread computes the machine schedule independently.
5. The main thread picks the bottleneck machine.
6. The bottleneck is fixed.
7. Already-fixed machines are re-optimized.
8. Repeat until done.

The important thing to understand is that the threads do **not** all edit the graph at once.
Only the evaluation step is parallel.
The graph updates still happen one at a time.

---

## 8. How to read the code in order

If you want to follow the program without getting lost, read it in this order:

1. `load_input()`
2. `build_graph()`
3. `compute_release()`
4. `compute_tails()`
5. `build_machine_problem()`
6. `schrage()`
7. `fix_machine()`
8. `shifting_bottleneck()` or `shifting_bottleneck_parallel()`
9. `extract_schedule()`
10. `write_output()`

That is the actual life cycle of the program.

---

## 9. Very short summary

The code is doing this:

- read a job-shop instance
- build a graph of operations
- repeatedly find the most critical machine
- schedule that machine
- update the graph
- repeat until all machines are fixed
- output the final schedule

The sequential file does this one machine at a time.
The parallel file does the machine evaluation step in parallel.
