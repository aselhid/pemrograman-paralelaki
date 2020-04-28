#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <string.h>

void sort(int *arr, int left, int right)
{
    if (left == right || left > right)
    {
        return;
    }

    int midpoint = left + (right - left) / 2;
    if (left < midpoint)
    {
        sort(arr, left, midpoint);
    }

    if (midpoint + 1 < right)
    {
        sort(arr, midpoint + 1, right);
    }

    int len = (right - left + 1);
    int *tmp_arr = malloc(len * sizeof(int));

    int l = left, r = midpoint + 1, idx = 0;
    while (l <= midpoint && r <= right)
    {
        if (arr[l] < arr[r])
        {
            tmp_arr[idx++] = arr[l++];
        }
        else
        {
            tmp_arr[idx++] = arr[r++];
        }
    }

    while (l <= midpoint)
    {
        tmp_arr[idx++] = arr[l++];
    }

    while (r <= right)
    {
        tmp_arr[idx++] = arr[r++];
    }

    idx = 0;
    for (int i = left; i <= right; i++)
    {
        arr[i] = tmp_arr[idx++];
    }

    free(tmp_arr);
}

int MIN_PARALLEL_LENGTH = 10000000; // 1e7

void parallel_sort(int *arr, int left, int right, int rootrank, MPI_Comm parent, int propagate, char *spawn_cmd)
{
    int len = (right - left + 1);
    if (left == right || left > right)
    {
        MPI_Send(arr, len, MPI_INT, 0, 3, parent);
        return;
    }

    int midpoint = left + (right - left) / 2;
    int left_len = midpoint - left + 1;
    int right_len = right - midpoint;

    MPI_Comm left_comm, right_comm;

    MPI_Request left_request = MPI_REQUEST_NULL, right_request = MPI_REQUEST_NULL, a, b, c, d;

    if (left_len > 0)
    {
        if (len <= MIN_PARALLEL_LENGTH)
        {
            sort(arr, left, midpoint);
        }
        else
        {
            int errcode;
            MPI_Comm_spawn(spawn_cmd, MPI_ARGV_NULL, 1, MPI_INFO_NULL, rootrank, MPI_COMM_SELF, &left_comm, &errcode);
            if (errcode != MPI_SUCCESS)
            {
                printf("failed spawning child");
                exit(1);
            }

            MPI_Irecv(arr, left_len, MPI_INT, 0, 3, left_comm, &left_request);
            MPI_Send(&left_len, 1, MPI_INT, 0, 1, left_comm);
            MPI_Isend(arr, left_len, MPI_INT, 0, 2, left_comm, &a);
        }
    }

    if (right_len > 0)
    {
        if (len <= MIN_PARALLEL_LENGTH)
        {
            sort(arr, midpoint + 1, right);
        }
        else
        {
            int errcode;
            MPI_Comm_spawn(spawn_cmd, MPI_ARGV_NULL, 1, MPI_INFO_NULL, rootrank, MPI_COMM_SELF, &right_comm, &errcode);
            if (errcode != MPI_SUCCESS)
            {
                printf("failed spawning child");
                exit(1);
            }

            MPI_Irecv(arr + midpoint + 1, right_len, MPI_INT, 0, 3, right_comm, &right_request);
            MPI_Send(&right_len, 1, MPI_INT, 0, 1, right_comm);
            MPI_Isend(arr + midpoint + 1, right_len, MPI_INT, 0, 2, right_comm, &b);
        }
    }

    MPI_Status lstatus, rstatus;
    if (left_request != MPI_REQUEST_NULL)
    {
        MPI_Wait(&left_request, &lstatus);
    }

    if (right_request != MPI_REQUEST_NULL)
    {
        MPI_Wait(&right_request, &rstatus);
    }

    int *tmp_arr = malloc(len * sizeof(int));

    int l = left, r = midpoint + 1, idx = 0;
    while (l <= midpoint && r <= right)
    {
        if (arr[l] < arr[r])
        {
            tmp_arr[idx++] = arr[l++];
        }
        else
        {
            tmp_arr[idx++] = arr[r++];
        }
    }

    while (l <= midpoint)
    {
        tmp_arr[idx++] = arr[l++];
    }

    while (r <= right)
    {
        tmp_arr[idx++] = arr[r++];
    }

    idx = 0;
    for (int i = left; i <= right; i++)
    {
        arr[i] = tmp_arr[idx++];
    }

    free(tmp_arr);

    if (propagate)
    {
        MPI_Send(arr, len, MPI_INT, 0, 3, parent);
    }
}

int main(int argc, char **argv)
{
    int rank;
    MPI_Comm parent;

    MPI_Init(&argc, &argv);
    MPI_Comm_get_parent(&parent);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (parent == MPI_COMM_NULL)
    {
        int arr_size = atoi(argv[1]);
        int *arr = malloc(arr_size * sizeof(int));

        for (int i = 0; i < arr_size; i++)
        {
            arr[i] = arr_size - i;
        }

        parallel_sort(arr, 0, arr_size - 1, rank, parent, 0, argv[0]);
    }
    else
    {
        int length;
        MPI_Recv(&length, 1, MPI_INT, 0, 1, parent, MPI_STATUS_IGNORE);

        int *arr = malloc(length * sizeof(int));
        MPI_Recv(arr, length, MPI_INT, 0, 2, parent, MPI_STATUS_IGNORE);
        parallel_sort(arr, 0, length - 1, rank, parent, 1, argv[0]);
    }
    MPI_Finalize();
    return 0;
}
