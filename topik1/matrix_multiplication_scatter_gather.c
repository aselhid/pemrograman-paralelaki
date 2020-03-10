#include "mpi.h"
#include <mpi.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define N 1024

double a[N][N], b[N][N], c[N][N];

void init_matrices()
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1.0;
            b[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

void validate_matrix()
{
    int i, j;
    bool valid;

    valid = true;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            valid &= (a[i][j] == c[i][j]);
        }
    }

    printf(valid ? "VALID," : "NOT,");
}

void print_matrix(double matrix[N][N])
{
    int i, j;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void main(int argc, char **argv)
{
    int numtasks, taskid, i, j, k, sum, count;
    float scattertime, bcasttime, gathertime;
    float avg_scattertime, avg_bcasttime, avg_gathertime;
    bool valid;

    struct timeval startjob, finishjob;
    struct timeval startscatter, finishscatter;
    struct timeval startbcast, finishbcast;
    struct timeval startgather, finishgather;

    init_matrices();

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Barrier(MPI_COMM_WORLD);

    count = N * N / numtasks;

    double atemp[N / numtasks][N], ctemp[N / numtasks][N];

    gettimeofday(&startjob, 0);

    gettimeofday(&startscatter, 0);
    MPI_Scatter(&a, count, MPI_DOUBLE, &atemp, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gettimeofday(&finishscatter, 0);

    gettimeofday(&startbcast, 0);
    MPI_Bcast(&b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gettimeofday(&finishbcast, 0);

    sum = 0;
    for (i = 0; i < N / numtasks; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < N; k++)
            {
                sum += atemp[i][k] * b[k][j];
            }
            ctemp[i][j] = sum;
            sum = 0;
        }
    }

    gettimeofday(&startgather, 0);
    MPI_Gather(&ctemp, count, MPI_DOUBLE, &c, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gettimeofday(&finishgather, 0);

    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&finishjob, 0);

    scattertime = (finishscatter.tv_sec + finishscatter.tv_usec * 1e-6) - (startscatter.tv_sec + startscatter.tv_usec * 1e-6);
    bcasttime = (finishbcast.tv_sec + finishbcast.tv_usec * 1e-6) - (startbcast.tv_sec + startbcast.tv_usec * 1e-6);
    gathertime = (finishgather.tv_sec + finishgather.tv_usec * 1e-6) - (startgather.tv_sec + startgather.tv_usec * 1e-6);

    MPI_Reduce(&scattertime, &avg_scattertime, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&bcasttime, &avg_bcasttime, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&gathertime, &avg_gathertime, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (taskid == 0)
    {
        avg_scattertime /= numtasks;
        avg_bcasttime /= numtasks;
        avg_gathertime /= numtasks;

        printf("%d,%d,", numtasks, N);
        validate_matrix();
        printf("%.6f,", (finishjob.tv_sec + finishjob.tv_usec * 1e-6) - (startjob.tv_sec + startjob.tv_usec * 1e-6));
        printf("%.6f,%.6f,", avg_scattertime + avg_bcasttime, avg_gathertime);
        printf("\n");
    }

    MPI_Finalize();
}
