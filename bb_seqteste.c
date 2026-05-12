#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#define MAX_JOBS 20
#define MAX_MACHINES 20
#define MAX_STACK 50000
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

static Node stack[MAX_STACK];
static int stackTop = 0;

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

int push(Node node) {
    if (stackTop >= MAX_STACK) { stackOverflow = 1; return 0; }
    stack[stackTop++] = node;
    return 1;
}

int pop(Node *node) {
    if (stackTop == 0) return 0;
    *node = stack[--stackTop];
    return 1;
}

void save_best(Node *node) {
    bestMakespan = node->makespan;
    for (int j = 0; j < nJobs; j++)
        for (int op = 0; op < nMachines; op++)
            bestStart[j][op] = node->startTime[j][op];
}

void reset_state() {
    bestMakespan = INT_MAX;
    nodesExplored = 0;
    stackOverflow = 0;
    stackTop = 0;
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

void branch_and_bound() {
    Node root;
    memset(&root, 0, sizeof(Node));
    root.lowerBound = calculate_lower_bound(&root);
    push(root);

    Node current;
    while (pop(&current)) {
        nodesExplored++;
        if (nodesExplored >= MAX_NODES_EXPLORED) break;
        if (current.lowerBound >= bestMakespan) continue;

        if (current.scheduled == nJobs * nMachines) {
            if (current.makespan < bestMakespan) save_best(&current);
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
            child.makespan    = maxInt(child.makespan, finish);
            child.lowerBound  = calculate_lower_bound(&child);

            if (child.lowerBound < bestMakespan) push(child);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s input.jss output.txt\n", argv[0]);
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

    double tempoTotal = 0.0;
    int    melhorMakespanFinal = INT_MAX;
    int    melhorStartFinal[MAX_JOBS][MAX_MACHINES];
    long long nosExploradosFinal = 0;
    int    stackOverflowFinal    = 0;

    for (int rep = 0; rep < NUM_REPETICOES; rep++) {
        reset_state();

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        greedy_initial_solution();
        branch_and_bound();

        clock_gettime(CLOCK_MONOTONIC, &ts1);

        double tempo = (ts1.tv_sec  - ts0.tv_sec)
                     + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
        tempoTotal += tempo;

        // Guardar a melhor solucao encontrada em qualquer repeticao 
        if (bestMakespan < melhorMakespanFinal) {
            melhorMakespanFinal = bestMakespan;
            for (int j = 0; j < nJobs; j++)
                for (int op = 0; op < nMachines; op++)
                    melhorStartFinal[j][op] = bestStart[j][op];
        }

        nosExploradosFinal = nodesExplored;
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
    printf("Nos explorados (ultima repeticao): %lld\n", nosExploradosFinal);
    printf("Repeticoes: %d\n", NUM_REPETICOES);
    printf("Tempo total (%d rep): %.6f segundos\n", NUM_REPETICOES, tempoTotal);
    printf("Tempo medio por repeticao: %.6f segundos\n", tempoMedio);

    if (nosExploradosFinal >= MAX_NODES_EXPLORED)
        printf("Aviso: limite de nos atingido. Resultado pode nao ser otimo.\n");
    if (stackOverflowFinal)
        printf("Aviso: stack encheu. Aumenta MAX_STACK.\n");

    return 0;
}
