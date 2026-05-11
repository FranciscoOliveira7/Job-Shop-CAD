#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

/*
 * Um lock por máquina:
 *   - Dois jobs que usam máquinas DIFERENTES são escalonados
 *     verdadeiramente em paralelo (sem espera).
 *   - Dois jobs que usam a MESMA máquina na mesma fase serializam
 *     apenas o acesso a essa máquina — o mínimo necessário para
 *     garantir correção sem condição de corrida.
 *
 * job_available[j] não precisa de lock porque cada posição j
 * é escrita/lida por exatamente um job (índice fixo por thread).
 */
omp_lock_t machine_locks[MAX_MACHINES];

/* ------------------------------------------------------------------ */
static inline int imax(int a, int b) { return a > b ? a : b; }

/* ------------------------------------------------------------------ */
/* Correção do makespan: o atomic de max acima pode não ser suportado */
/* em todos os compiladores. Usamos esta versão alternativa segura:   */
/* ------------------------------------------------------------------ */
static int schedule_parallel(int num_jobs, int num_machines, int num_threads)
{
    int num_operations = num_machines;

    for (int j = 0; j < num_jobs; j++)
        job_available[j] = 0;
    for (int m = 0; m < num_machines; m++)
        machine_available[m] = 0;

    int makespan = 0;

#pragma omp parallel num_threads(num_threads) shared(jobs, job_available, machine_available, makespan)
    {
        for (int o = 0; o < num_operations; o++)
        {

#pragma omp for schedule(static)
            for (int j = 0; j < num_jobs; j++)
            {

                int m = jobs[j][o].machine;
                int duration = jobs[j][o].duration;

                int ja = job_available[j]; /* leitura privada — sem lock */

                omp_set_lock(&machine_locks[m]);

                int t_start = imax(ja, machine_available[m]);
                int t_finish = t_start + duration;

                jobs[j][o].start = t_start;
                machine_available[m] = t_finish;

                omp_unset_lock(&machine_locks[m]);

                job_available[j] = t_finish; /* escrita privada — sem lock */

/* Atualização segura do makespan global */
#pragma omp critical(makespan_update)
                {
                    if (t_finish > makespan)
                        makespan = t_finish;
                }
            }
            /* barreira implícita aqui */
        }
    }

    return makespan;
}

/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Uso: %s ficheiro_entrada ficheiro_saida num_threads\n", argv[0]);
        return 1;
    }

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    int num_threads = atoi(argv[3]);

    if (num_threads < 1)
    {
        fprintf(stderr, "Numero de threads invalido: %d\n", num_threads);
        return 1;
    }

    /* ---- Leitura do ficheiro de entrada ---- */
    FILE *fin = fopen(input_filename, "r");
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

    /* ---- Inicialização dos locks por máquina ---- */
    for (int m = 0; m < num_machines; m++)
        omp_init_lock(&machine_locks[m]);

    /* ---- Medição do tempo (NUM_REPEATS repetições) ---- */
    double total_elapsed = 0.0;
    int makespan = 0;

    for (int r = 0; r < NUM_REPEATS; r++)
    {
        /*
         * O tempo inclui criação e término das threads (omp parallel),
         * conforme exigido pelo enunciado.
         */
        double t0 = omp_get_wtime();
        makespan = schedule_parallel(num_jobs, num_machines, num_threads);
        double t1 = omp_get_wtime();

        total_elapsed += (t1 - t0);
    }

    double avg_elapsed = total_elapsed / NUM_REPEATS;

    /* ---- Destruição dos locks ---- */
    for (int m = 0; m < num_machines; m++)
        omp_destroy_lock(&machine_locks[m]);

    /* ---- Tempo para a consola (nunca para o ficheiro de saída) ---- */
    fprintf(stdout, "Threads: %d | Tempo medio (%d repeticoes): %.6f s | Makespan: %d\n",
            num_threads, NUM_REPEATS, avg_elapsed, makespan);

    /* ---- Escrita do ficheiro de saída (formato exato do enunciado) ---- */
    FILE *fout = fopen(output_filename, "w");
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