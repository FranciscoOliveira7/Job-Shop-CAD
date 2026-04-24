#include <stdio.h>
#include <stdlib.h>

// OS specific stuff
#ifdef _WIN32
#include <windows.h>
#endif

#ifdef linux
#include <unistd.h>
#endif

#define JOBS 3
#define OPS 3

int spent_time = 0;

void wait_seconds(int duration);
void print_matrix(int matrix[3][3]);

void operate(int input[3][6], int output_matrix[3][3], int job, int operation) {
    int machine_index = input[job][operation * 2];
    int duration = input[job][operation * 2 + 1];

    wait_seconds(duration);
    printf("Machine %d finished op %d\n", machine_index, operation);
    output_matrix[job][operation] = spent_time;

    spent_time += duration;
}

int main() {
    printf("Hello world!\n");

    // to see if it's running or not
    // int machines_state[3] = { 0, 0, 0 };

    // 3 jobs, 3 operations, 2 (machine, duration)
    int test[JOBS][OPS * 2] = {
        { 0, 3, 1, 2, 2, 2 },
        { 0, 2, 2, 1, 1, 4 },
        { 1, 4, 2, 3, 0, 1 }
    };
    int output[JOBS][OPS] = {
        { -1, -1, -1 },
        { -1, -1, -1 },
        { -1, -1, -1 }
    };

    int count = JOBS * OPS;

    // I'm cooking, I swear...

    // Extremely complex algorithm
    for (int i = 0; i < OPS; i++) {
        for (int j = 0; j < JOBS; j++) {
            operate(test, output, j, i);
            count--;
        }
    }

    printf("-- Output --\n");
    print_matrix(output);

    return 0;
}

void print_matrix(int matrix[3][3]) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf(" %2d", matrix[i][j]);
        }
        printf("\n");
    }
}

void wait_seconds(int duration) {
    #ifdef _WIN32
    sleep(duration * 1000);
    #endif

    #ifdef linux
    sleep(duration);
    #endif
}
