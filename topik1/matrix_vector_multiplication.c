/**********************************************************************                                                                                      
 * MPI-based matrix multiplication AxB=C                                                                                                                     
 *********************************************************************/

#include "mpi.h"
#define N 2048 /* number of rows and columns in matrix */
#define M 2048 /* number of rows and columns in matrix */
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
MPI_Status status;

double a[M][N], b[N], c[M];

main(int argc, char **argv)
{
    int numtasks, taskid, numworkers, source, dest, rows, offset, i, j, k;

    struct timeval startjob, finishjob;
    struct timeval startcomm, finishcomm;
    struct timeval startfanin;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks - 1;

    /*---------------------------- master ----------------------------*/
    if (taskid == 0)
    {
        for (i = 0; i < M; i++)
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

        gettimeofday(&startjob, 0);

        /* send matrix data to the worker tasks */
        rows = N / numworkers;
        offset = 0;

        gettimeofday(&startcomm, 0);
        for (dest = 1; dest <= numworkers; dest++)
        {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&b, M, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset = offset + rows;
        }
        gettimeofday(&finishcomm, 0);

        gettimeofday(&startfanin, 0);
        /* wait for results from all worker tasks */
        for (i = 1; i <= numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset], rows, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        gettimeofday(&finishjob, 0);

        bool valid = true;
        for (i = 0; i < N; i++)
        {
            // printf("%6.2f   ", c[i]);
            if (c[i] != N)
            {
                valid = false;
                break;
            }
        }

        printf("%d,%d,%d,", numtasks, N, M);
        printf(valid ? "VALID," : "NOT,");
        printf("%.6f,", (finishjob.tv_sec + finishjob.tv_usec * 1e-6) - (startjob.tv_sec + startjob.tv_usec * 1e-6));
        printf("%.6f,", (finishcomm.tv_sec + finishcomm.tv_usec * 1e-6) - (startcomm.tv_sec + startcomm.tv_usec * 1e-6));
        printf("%.6f,", (finishjob.tv_sec + finishjob.tv_usec * 1e-6) - (startfanin.tv_sec + startfanin.tv_usec * 1e-6));
        printf("\n");
    }

    /*---------------------------- worker----------------------------*/
    if (taskid > 0)
    {
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, N * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        /* Matrix multiplication */
        for (k = 0; k < N; k++)
            for (i = 0; i < rows; i++)
            {
                c[i] = 0.0;
                for (j = 0; j < N; j++)
                    c[i] = c[i] + a[i][j] * b[j];
            }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c, rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
