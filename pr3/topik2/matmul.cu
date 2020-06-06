#include <cstdio>

#define BLOCK_SIZE 32

__global__ void matrix_multiplication_square(float *d_a, float *d_b, float *d_c, int n) {
    __shared__ float a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; sub++) {
        idx = row * n + sub * BLOCK_SIZE  + threadIdx.x;

        if (idx >= n * n) {
            a[threadIdx.y][threadIdx.x] = 0;
        } else {
            a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        
        if(idx >= n * n) {
            b[threadIdx.y][threadIdx.x] = 0;
        } else {
            b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            tmp += a[threadIdx.y][k] * b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        d_c[row * n + col] = tmp;
    }
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);

    float *h_a, *h_b, *h_c;
    int nbytes = n * n * sizeof(float);

    cudaMallocHost((void **) &h_a, nbytes);
    cudaMallocHost((void **) &h_b, nbytes);
    cudaMallocHost((void **) &h_c, nbytes);

    // init matrix
    cudaMemset(h_b, 0, nbytes);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_a[i * n + j] = 1;

            if (i == j) {
                h_b[i * n + j] = 1;
            }
        }
    }

    float gpu_elapsed_time_ms;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // memory on the device
    cudaEventRecord(start, 0);
    float *d_a, *d_b, *d_c;

    cudaMallocHost((void **) &d_a, nbytes);
    cudaMallocHost((void **) &d_b, nbytes);
    cudaMallocHost((void **) &d_c, nbytes);

    // copy matrix to device
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nbytes, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    matrix_multiplication_square<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, nbytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
 //   printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", n, n, n, n, gpu_elapsed_time_ms);

    bool good_answer = true;

    for (int i = 0; i < n * n; i++) {
        good_answer &= (d_a[i] == d_c[i]);
    }

    if (good_answer) {
        printf("Good answer\n");
    } else {
        printf("Bad answer\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_a);
    cudaFreeHost(h_a);

    return 0;
}
