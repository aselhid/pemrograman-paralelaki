#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cstdio>

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);

    float *hA = (float*) malloc(N * N * sizeof(float));
    float *hB = (float*) malloc(N * N * sizeof(float));
    float *hC = (float*) malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            hA[i*N+j] = 1;
            hB[i*N+j] = i == j ? 1 : 0;
        }
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, N * N * sizeof(float));
    cudaMalloc(&dB, N * N * sizeof(float));
    cudaMalloc(&dC, N * N * sizeof(float));
    
    cudaMemcpy(dA, hA, N * N * sizeof(float), cudaMemcpyHostToDevice);    
    cudaMemcpy(dB, hB, N * N * sizeof(float), cudaMemcpyHostToDevice);    

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1, beta = 0;
    cublasGemmEx(
	handle,
	CUBLAS_OP_N,
	CUBLAS_OP_N,
	N, N, N,
	&alpha,
	dA, CUDA_R_32F, N,
	dB, CUDA_R_32F, N,
	&beta,
	dC, CUDA_R_32F, N,
	CUDA_R_32F,
	CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpy(hC, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(hC[i*N+j] == 1);
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hA);
    free(hB);
    free(hC);
}

