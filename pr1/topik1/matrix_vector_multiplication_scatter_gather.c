#include "mpi.h"
#include <mpi.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define N 1024
#define M 1024

double a[M][N], b[N], c[M];
double atemp[M][N], ctemp[M];

void init_matrices()
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            a[i][j] = 1.0;
        }
    }
    for (i = 0; i < N; i++)
    {
        b[i] = 1.0;
    }
}

void validate_vector()
{
    int i;
    bool valid;

    valid = true;
    for (i = 0; i < N; i++)
    {
        if (c[i] != N)
        {
            valid = false;
            break;
        }
    }

    printf(valid ? "VALID," : "NOT,");
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

    count = N * M / numtasks;

    gettimeofday(&startjob, 0);

    gettimeofday(&startscatter, 0);
    MPI_Scatter(&a, count, MPI_DOUBLE, &atemp, count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gettimeofday(&finishscatter, 0);

    gettimeofday(&startbcast, 0);
    MPI_Bcast(&b, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    gettimeofday(&finishbcast, 0);

    sum = 0;
    for (i = 0; i < M / numtasks; i++)
    {
        for (j = 0; j < N; j++)
        {
            for (k = 0; k < M; k++)
            {
                sum += atemp[i][k] * b[j];
            }
            ctemp[i] = sum;
            sum = 0;
        }
    }

    gettimeofday(&startgather, 0);
    MPI_Gather(&ctemp, M / numtasks, MPI_DOUBLE, &c, M / numtasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
        validate_vector();
        printf("%.6f,", (finishjob.tv_sec + finishjob.tv_usec * 1e-6) - (startjob.tv_sec + startjob.tv_usec * 1e-6));
        printf("%.6f,%.6f,", avg_scattertime + avg_bcasttime, avg_gathertime);
        printf("\n");
    }

    MPI_Finalize();
}
