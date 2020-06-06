#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cstdio>

__global__ void multiply(float *a, float *b, float *c, int N) {
    int n = blockIdx.x * 1024 + threadIdx.x;
    int i = n / N;
    int j = n % N;
  
    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum = sum + a[i*N + k] * b[k*N + j];
    }

    c[i*N + j] = sum;
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

    int numBlock = ceil(N*N / 1024.0);
    int numThread = min(N*N, 1024);
    multiply<<<numBlock, 1024>>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(abs(hC[i*N+j] - (i*N+j + 1)) < 1e-6);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hC);
}
