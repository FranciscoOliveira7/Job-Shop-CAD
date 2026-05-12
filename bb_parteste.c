#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define MAX_JOBS 20
#define MAX_MACHINES 20
#define MAX_INITIAL_NODES 10000
#define MAX_LOCAL_STACK 30000
#define MAX_NODES_EXPLORED 34000000LL

#define NUM_REPETICOES 1   

static int nJobs, nMachines;
static int machine[MAX_JOBS][MAX_MACHINES];
static int duration[MAX_JOBS][MAX_MACHINES];

typedef struct {
    int nextOp[MAX_JOBS];
    int jobReady[MAX_JOBS];
    int machineReady[MAX_MACHINES];
    int startTime[MAX_JOBS][MAX_MACHINES];
    int scheduled;
    int makespan;
    int lowerBound;
} Node;

static Node initialNodes[MAX_INITIAL_NODES];
static int nInitialNodes = 0;

static int bestMakespan = INT_MAX;
static int bestStart[MAX_JOBS][MAX_MACHINES];

static long long nodesExplored = 0;
static int stackOverflow = 0;

int maxInt(int a, int b) {
    return a > b ? a : b;
}

int remaining_time_job(Node *node, int j) {
    int sum = 0;
    for (int op = node->nextOp[j]; op < nMachines; op++)
        sum += duration[j][op];
    return sum;
}

int calculate_lower_bound(Node *node) {
    int lb = node->makespan;

    for (int j = 0; j < nJobs; j++) {
        int finish = node->jobReady[j] + remaining_time_job(node, j);
        if (finish > lb) lb = finish;
    }

    for (int m = 0; m < nMachines; m++) {
        int load = 0;
        for (int j = 0; j < nJobs; j++)
            for (int op = node->nextOp[j]; op < nMachines; op++)
                if (machine[j][op] == m)
                    load += duration[j][op];

        int finishMachine = node->machineReady[m] + load;
        if (finishMachine > lb) lb = finishMachine;
    }

    return lb;
}

void save_best(Node *node) {
    bestMakespan = node->makespan;
    for (int j = 0; j < nJobs; j++)
        for (int op = 0; op < nMachines; op++)
            bestStart[j][op] = node->startTime[j][op];
}

void reset_state() {
    bestMakespan  = INT_MAX;
    nodesExplored = 0;
    stackOverflow = 0;
    nInitialNodes = 0;
    memset(bestStart, 0, sizeof(bestStart));
}

void greedy_initial_solution() {
    Node node;
    memset(&node, 0, sizeof(Node));

    for (int op = 0; op < nMachines; op++) {
        for (int j = 0; j < nJobs; j++) {
            int m = machine[j][op];
            int d = duration[j][op];
            int start  = maxInt(node.jobReady[j], node.machineReady[m]);
            int finish = start + d;
            node.startTime[j][op] = start;
            node.jobReady[j]      = finish;
            node.machineReady[m]  = finish;
            node.nextOp[j]++;
            node.scheduled++;
            node.makespan = maxInt(node.makespan, finish);
        }
    }

    save_best(&node);
}

void add_initial_node(Node node) {
    if (nInitialNodes < MAX_INITIAL_NODES)
        initialNodes[nInitialNodes++] = node;
}

void generate_initial_nodes_rec(Node current, int depth, int maxDepth) {
    if (nInitialNodes >= MAX_INITIAL_NODES) return;

    current.lowerBound = calculate_lower_bound(&current);
    if (current.lowerBound >= bestMakespan) return;

    if (depth == maxDepth || current.scheduled == nJobs * nMachines) {
        add_initial_node(current);
        return;
    }

    for (int j = 0; j < nJobs; j++) {
        int op = current.nextOp[j];
        if (op >= nMachines) continue;

        Node child = current;
        int m = machine[j][op];
        int d = duration[j][op];
        int start  = maxInt(child.jobReady[j], child.machineReady[m]);
        int finish = start + d;
        child.startTime[j][op] = start;
        child.jobReady[j]      = finish;
        child.machineReady[m]  = finish;
        child.nextOp[j]++;
        child.scheduled++;
        child.makespan    = maxInt(child.makespan, finish);
        child.lowerBound  = calculate_lower_bound(&child);

        if (child.lowerBound < bestMakespan)
            generate_initial_nodes_rec(child, depth + 1, maxDepth);
    }
}

void generate_initial_nodes(int depth) {
    Node root;
    memset(&root, 0, sizeof(Node));
    root.lowerBound = calculate_lower_bound(&root);
    generate_initial_nodes_rec(root, 0, depth);
    if (nInitialNodes == 0) add_initial_node(root);
}

void explore_subtree(Node startNode) {
    Node *stack = malloc(MAX_LOCAL_STACK * sizeof(Node));
    if (stack == NULL) { printf("Erro ao alocar stack local.\n"); exit(1); }

    int top = 0;
    stack[top++] = startNode;

    while (top > 0) {
        long long currentNodes;

        #pragma omp critical
        {
            currentNodes = ++nodesExplored;
        }

        if (currentNodes >= MAX_NODES_EXPLORED) break;

        Node current = stack[--top];

        int localBest;
        #pragma omp critical
        {
            localBest = bestMakespan;
        }

        if (current.lowerBound >= localBest) continue;

        if (current.scheduled == nJobs * nMachines) {
            if (current.makespan < localBest) {
                #pragma omp critical
                {
                    if (current.makespan < bestMakespan)
                        save_best(&current);
                }
            }
            continue;
        }

        for (int j = nJobs - 1; j >= 0; j--) {
            int op = current.nextOp[j];
            if (op >= nMachines) continue;

            Node child = current;
            int m = machine[j][op];
            int d = duration[j][op];
            int start  = maxInt(child.jobReady[j], child.machineReady[m]);
            int finish = start + d;
            child.startTime[j][op] = start;
            child.jobReady[j]      = finish;
            child.machineReady[m]  = finish;
            child.nextOp[j]++;
            child.scheduled++;
            child.makespan   = maxInt(child.makespan, finish);
            child.lowerBound = calculate_lower_bound(&child);

            #pragma omp critical
            {
                localBest = bestMakespan;
            }

            if (child.lowerBound < localBest) {
                if (top < MAX_LOCAL_STACK) {
                    stack[top++] = child;
                } else {
                    #pragma omp critical
                    stackOverflow = 1;
                }
            }
        }
    }

    free(stack);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Uso: %s input.jss output.txt num_threads\n", argv[0]);
        return 1;
    }

    FILE *fin = fopen(argv[1], "r");
    if (fin == NULL) { perror("Erro ao abrir ficheiro de entrada"); return 1; }

    if (fscanf(fin, "%d %d", &nJobs, &nMachines) != 2) {
        printf("Erro no formato do ficheiro.\n");
        fclose(fin); return 1;
    }

    if (nJobs > MAX_JOBS || nMachines > MAX_MACHINES) {
        printf("Erro: problema demasiado grande para estes limites.\n");
        fclose(fin); return 1;
    }

    for (int j = 0; j < nJobs; j++)
        for (int op = 0; op < nMachines; op++)
            if (fscanf(fin, "%d %d", &machine[j][op], &duration[j][op]) != 2) {
                printf("Erro ao ler job %d operacao %d.\n", j, op);
                fclose(fin); return 1;
            }
    fclose(fin);

    int numThreads = atoi(argv[3]);
    if (numThreads <= 0) { printf("Erro: numero de threads invalido.\n"); return 1; }

    omp_set_num_threads(numThreads);

    double tempoTotal = 0.0;
    int    melhorMakespanFinal = INT_MAX;
    int    melhorStartFinal[MAX_JOBS][MAX_MACHINES];
    long long nosExploradosFinal = 0;
    int    nosIniciaisFinal      = 0;
    int    stackOverflowFinal    = 0;

    for (int rep = 0; rep < NUM_REPETICOES; rep++) {
        reset_state();

        double t0 = omp_get_wtime(); 

        greedy_initial_solution();
        generate_initial_nodes(4);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nInitialNodes; i++)
            explore_subtree(initialNodes[i]);

        double t1 = omp_get_wtime();

        tempoTotal += (t1 - t0);

        // Guardar a melhor solucao encontrada em qualquer repeticao 
        if (bestMakespan < melhorMakespanFinal) {
            melhorMakespanFinal = bestMakespan;
            for (int j = 0; j < nJobs; j++)
                for (int op = 0; op < nMachines; op++)
                    melhorStartFinal[j][op] = bestStart[j][op];
        }

        nosExploradosFinal = nodesExplored;
        nosIniciaisFinal   = nInitialNodes;
        stackOverflowFinal |= stackOverflow;
    }

    double tempoMedio = tempoTotal / NUM_REPETICOES;

    FILE *fout = fopen(argv[2], "w");
    if (fout == NULL) { perror("Erro ao criar ficheiro de saida"); return 1; }

    fprintf(fout, "%d\n", melhorMakespanFinal);
    for (int j = 0; j < nJobs; j++) {
        for (int op = 0; op < nMachines; op++) {
            fprintf(fout, "%d", melhorStartFinal[j][op]);
            if (op < nMachines - 1) fprintf(fout, " ");
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    printf("Ficheiro criado: %s\n", argv[2]);
    printf("Melhor makespan: %d\n", melhorMakespanFinal);
    printf("Nos iniciais: %d\n", nosIniciaisFinal);
    printf("Nos explorados (ultima repeticao): %lld\n", nosExploradosFinal);
    printf("Threads: %d\n", numThreads);
    printf("Repeticoes: %d\n", NUM_REPETICOES);
    printf("Tempo total (%d rep): %.6f segundos\n", NUM_REPETICOES, tempoTotal);
    printf("Tempo medio por repeticao: %.6f segundos\n", tempoMedio);

    if (nosExploradosFinal >= MAX_NODES_EXPLORED)
        printf("Aviso: limite de nos atingido. Resultado pode nao ser otimo.\n");
    if (stackOverflowFinal)
        printf("Aviso: stack local encheu. Aumenta MAX_LOCAL_STACK.\n");

    return 0;
}
