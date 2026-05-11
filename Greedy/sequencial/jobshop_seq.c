#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_JOBS 200
#define MAX_MACHINES 200
#define MAX_OPS 200
#define NUM_REPEATS 10 /* número de repetições para medir tempo */

typedef struct
{
    int machine;
    int duration;
    int start;
} Operation;

/* ------------------------------------------------------------------ */
/* Estruturas globais (arrays estáticos, sem apontadores internos)     */
/* ------------------------------------------------------------------ */
Operation jobs[MAX_JOBS][MAX_OPS];

/* Instante em que cada job fica disponível para a próxima operação   */
int job_available[MAX_JOBS];

/* Instante em que cada máquina fica livre                            */
int machine_available[MAX_MACHINES];

/* ------------------------------------------------------------------ */
/* Utilitários                                                         */
/* ------------------------------------------------------------------ */
static inline int imax(int a, int b) { return a > b ? a : b; }

/* ------------------------------------------------------------------ */
/* Algoritmo greedy sequencial                                         */
/*                                                                     */
/* Para cada operação o (por ordem) de cada job j:                    */
/*   start = max(job_available[j], machine_available[machine])        */
/*   Atualiza job_available[j] e machine_available[machine].          */
/*                                                                     */
/* O loop externo é por operação para que, ao processar a operação o  */
/* do job j, a operação o-1 de j já esteja sempre escalonada.         */
/* Dentro de cada fase o, os jobs são processados por ordem crescente */
/* de j — escolha determinista que evita ambiguidade.                 */
/* ------------------------------------------------------------------ */
static int schedule(int num_jobs, int num_machines)
{
    int num_operations = num_machines; /* cada job tem exatamente num_machines operações */

    /* Reinicializa disponibilidades */
    for (int j = 0; j < num_jobs; j++)
        job_available[j] = 0;
    for (int m = 0; m < num_machines; m++)
        machine_available[m] = 0;

    int makespan = 0;

    for (int o = 0; o < num_operations; o++)
    {
        for (int j = 0; j < num_jobs; j++)
        {
            int m = jobs[j][o].machine;
            int duration = jobs[j][o].duration;

            int t_start = imax(job_available[j], machine_available[m]);
            int t_finish = t_start + duration;

            jobs[j][o].start = t_start;
            job_available[j] = t_finish;
            machine_available[m] = t_finish;

            if (t_finish > makespan)
                makespan = t_finish;
        }
    }
    return makespan;
}

/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Uso: %s ficheiro_entrada ficheiro_saida\n", argv[0]);
        return 1;
    }

    /* ---- Leitura do ficheiro de entrada ---- */
    FILE *fin = fopen(argv[1], "r");
    if (!fin)
    {
        perror("Erro ao abrir ficheiro de entrada");
        return 1;
    }

    int num_jobs, num_machines;
    fscanf(fin, "%d %d", &num_jobs, &num_machines);

    int num_operations = num_machines;

    for (int j = 0; j < num_jobs; j++)
        for (int o = 0; o < num_operations; o++)
            fscanf(fin, "%d %d", &jobs[j][o].machine, &jobs[j][o].duration);

    fclose(fin);

    /* ---- Medição do tempo (NUM_REPEATS repetições) ---- */
    struct timespec ts_start, ts_end;
    double total_elapsed = 0.0;
    int makespan = 0;

    for (int r = 0; r < NUM_REPEATS; r++)
    {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);
        makespan = schedule(num_jobs, num_machines);
        clock_gettime(CLOCK_MONOTONIC, &ts_end);

        double elapsed = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
        total_elapsed += elapsed;
    }

    double avg_elapsed = total_elapsed / NUM_REPEATS;

    /* Imprime tempo na consola (não no ficheiro de saída) */
    fprintf(stdout, "Tempo medio de execucao (%d repeticoes): %.6f s\n",
            NUM_REPEATS, avg_elapsed);

    /* ---- Escrita do ficheiro de saída ---- */
    FILE *fout = fopen(argv[2], "w");
    if (!fout)
    {
        perror("Erro ao abrir ficheiro de saida");
        return 1;
    }

    fprintf(fout, "%d\n", makespan);

    for (int j = 0; j < num_jobs; j++)
    {
        for (int o = 0; o < num_operations; o++)
        {
            if (o > 0)
                fprintf(fout, " ");
            fprintf(fout, "%d", jobs[j][o].start);
        }
        fprintf(fout, "\n");
    }

    fclose(fout);
    return 0;
}
