#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 1024  // Tamaño de las matrices cuadradas NxN

// Función para llenar una matriz con valores aleatorios entre 0 y 9
void fill_matrix(double* matrix) {
    for (int i = 0; i < N * N; i++)
        matrix[i] = rand() % 10;
}

int main(int argc, char* argv[]) {
    int rank, size;

    // Inicializa el entorno MPI y obtiene el rango y tamaño del comunicador
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Rango del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Número total de procesos

    // Punteros para matrices A, B y C (matrices NxN almacenadas linealmente)
    double* A = NULL;
    double* B = malloc(N * N * sizeof(double));  // Matriz B, será enviada a todos
    double* C = NULL;

    // Solo el proceso raíz (rank 0) inicializa matrices A y C y llena A y B con valores
    if (rank == 0) {
        A = malloc(N * N * sizeof(double));  // Matriz A completa
        C = malloc(N * N * sizeof(double));  // Resultado de la multiplicación
        fill_matrix(A);  // Inicializa A con valores aleatorios
        fill_matrix(B);  // Inicializa B con valores aleatorios
    }

    // Reservar espacio para la porción local de A y C para cada proceso
    // Cada proceso manejará N/size filas de la matriz A y el resultado parcial C
    double* local_A = malloc((N / size) * N * sizeof(double));
    double* local_C = malloc((N / size) * N * sizeof(double));

    // Distribuye la matriz B completa a todos los procesos
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Divide y reparte porciones de A entre los procesos
    // Cada proceso recibe N/size filas de A (con N columnas)
    MPI_Scatter(A, (N * N) / size, MPI_DOUBLE, local_A, (N * N) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Variables para bucles paralelos
    int i, j, k;

    // Multiplicación paralela híbrida: MPI distribuye filas, OpenMP paraleliza columnas
    #pragma omp parallel for private(j, k)
    for (i = 0; i < N / size; i++) {        // Cada proceso recorre sus filas asignadas
        for (j = 0; j < N; j++) {            // Para cada columna de B
            double sum = 0.0;
            for (k = 0; k < N; k++) {        // Producto escalar fila de A y columna de B
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;        // Guardar resultado parcial
        }
    }

    // Recolecta las partes calculadas localmente en la matriz C del proceso raíz
    MPI_Gather(local_C, (N * N) / size, MPI_DOUBLE, C, (N * N) / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Solo el proceso raíz imprime un valor para verificar el resultado
    if (rank == 0) {
        printf("C[0][0] = %f\n", C[0]);  // Mostrar primer elemento de C
        free(A);                         // Liberar memoria de A
        free(C);                         // Liberar memoria de C
    }

    // Liberar memoria en todos los procesos
    free(B);
    free(local_A);
    free(local_C);

    // Finaliza el entorno MPI
    MPI_Finalize();
    return 0;
}


