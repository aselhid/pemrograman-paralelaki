#include <stdio.h>
#include <cuda.h>

__global__ void run(int *a, int *b, int N) {
    int idx = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y + threadIdx.x + threadIdx.y * blockDim.y;

    if (idx < N) {
        a[idx] = threadIdx.x;
        b[idx] = blockIdx.x;
    }
}

int main() {
    int *a, *b, N = 30;
    cudaMalloc((void **)&a, N * sizeof(int));
    cudaMalloc((void **)&b, N * sizeof(int));
    
    dim3 gridSize(2, 2);
    dim3 blockSize(2, 2); 
    // run<<<gridSize, blockSize>>>(a, b, N);
    run<<<2, 4>>>(a, b, N);

    int *hA = new int[N], *hB = new int[N];
    cudaMemcpy(hA, a, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hB, b, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%d: %d[%d]\n", i, hA[i], hB[i]);
    }

    cudaFree(a);
    cudaFree(b);
}

