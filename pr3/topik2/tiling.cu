#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cstdio>

__global__ void multiply(float *a, float *b, float *c, int N) {
    __shared__ float sA[32][32];
    __shared__ float sB[32][32];
    
    int ii = blockIdx.y * blockDim.y + threadIdx.y; 
    int jj = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int i = 0; i < gridDim.x; i += 1) {
        int x = i * blockDim.x + threadIdx.x; 
        int y = i * blockDim.x + threadIdx.y; 

        sA[threadIdx.y][threadIdx.x] = a[ii*N + x];
        sB[threadIdx.x][threadIdx.y] = b[y*N + jj];

        __syncthreads();
        for (int k = 0; k < blockDim.x; k++) {
            sum = sum  + sA[threadIdx.y][k] * sB[threadIdx.x][k];
        }
        __syncthreads();
    }

    c[ii*N + jj] = sum;
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);

    float *hA = (float*) malloc(N * N * sizeof(float));
    float *hB = (float*) malloc(N * N * sizeof(float));
    float *hC = (float*) malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            hA[i*N+j] = i*N+j + 1;

            if (i == j) hB[i*N+j] = 1;
        }
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, N * N * sizeof(float));
    cudaMalloc(&dB, N * N * sizeof(float));
    cudaMalloc(&dC, N * N * sizeof(float));
    
    cudaMemcpy(dA, hA, N * N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(dB, hB, N * N * sizeof(float), cudaMemcpyHostToDevice);    
    int sz = min(N, 32);
    sz = 4;
    dim3 dimGrid(N / sz, N / sz);
    dim3 dimBlock(sz, sz);

    multiply<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(abs(hC[i*N+j] - (i*N+j + 1)) < 1e-9);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hC);
}
