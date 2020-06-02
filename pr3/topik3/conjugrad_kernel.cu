#include <cstdio>
#include <stdlib.h>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "conjugrad_kernel.cuh"

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

// to avoid device-host memory passing 
__global__ void scalar_div(double* a, double* b, double* res)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
		*res = (*a) / (*b);
}

__global__ void inverse_sign(double* a, double* res)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
		*res = -*a;
}

void print_device_variable(double* d_x, int n)
{
	double* x = (double*)malloc(sizeof(*x) * n);
	cudaMemcpy(x, d_x, sizeof(*x) * n, cudaMemcpyDeviceToHost);
	if (n > 1)
	{
		for (int i = 0; i < n; ++i)
			printf("%.17g ", x[i]);
		printf("\n");
	}
	else {
		printf("%.17g\n", *x);
	}
}

void gpu_conjugate_gradient(double* A, double* b, int n, conjugrad_params params, double* x)
{
	cublasHandle_t cublas_handle;

	double* d_A, * d_b, * d_x, * d_r, * d_rdot,  *d_rprevdot, * d_p, * d_vtmp, * d_tmp, * d_alpha, * d_beta, * scalar_r, *tmp;
	unsigned int iteration;
	
	size_t elem_size = sizeof(double);
	size_t vector_size = elem_size * n;
	size_t matrix_size = elem_size * n * n;

	scalar_r = (double*)malloc(sizeof(*scalar_r));
	tmp = (double*)malloc(sizeof(*tmp));

	// device memory allocation and device-host data movement
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_A, matrix_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_b, vector_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_x, vector_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_r, vector_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_p, vector_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_vtmp, vector_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_alpha, elem_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_beta, elem_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_rdot, elem_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_rprevdot, elem_size));
	CHECK_CUDA_ERR(cudaMalloc((void**)&d_tmp, elem_size));

	CHECK_CUBLAS_ERR(cublasCreate(&cublas_handle));
	CHECK_CUBLAS_ERR(cublasSetMatrix(n, n, elem_size, A, n, d_A, n));
	CHECK_CUBLAS_ERR(cublasSetVector(n, elem_size, b, 1, d_b, 1));

	// initial guess of x vector set to 0 vector
	CHECK_CUDA_ERR(cudaMemset(d_x, 0, vector_size));
	CHECK_CUBLAS_ERR(cublasDcopy_v2(cublas_handle, n, d_b, 1, d_r, 1));
	CHECK_CUBLAS_ERR(cublasDcopy_v2(cublas_handle, n, d_b, 1, d_p, 1));
	CHECK_CUBLAS_ERR(cublasDdot_v2(cublas_handle, n, d_r, 1, d_r, 1, d_rdot));

	iteration = 0;
	while (iteration < params.max_iteration)
	{
		// calculate alpha
		CHECK_CUBLAS_ERR(cublasDsymv_v2(cublas_handle, CUBLAS_FILL_MODE_LOWER, n, &ONE, d_A, n, d_p, 1, &ZERO, d_vtmp, 1));
		CHECK_CUBLAS_ERR(cublasDdot_v2(cublas_handle, n, d_vtmp, 1, d_p, 1, d_tmp));
		scalar_div<<<1, 1>>>(d_rdot, d_tmp, d_alpha);

		// calculate new x
		CHECK_CUDA_ERR(cudaMemcpy(tmp, d_alpha, elem_size, cudaMemcpyDeviceToHost)); // hack alpha cant be on device memory space
		CHECK_CUBLAS_ERR(cublasDaxpy_v2(cublas_handle, n, tmp, d_p, 1, d_x, 1));

		// store previous residue and dot
		// CHECK_CUBLAS_ERR(cublasDcopy(cublas_handle, n, d_r, 1, d_rprev, 1));
		CHECK_CUDA_ERR(cudaMemcpy(d_rprevdot, d_rdot, elem_size, cudaMemcpyDeviceToDevice));
		
		// calculate new residue and dot
		inverse_sign<<<1, 1>>>(d_alpha, d_tmp);
		CHECK_CUDA_ERR(cudaMemcpy(tmp, d_tmp, elem_size, cudaMemcpyDeviceToHost)); // hack alpha cant be on device memory space
		CHECK_CUBLAS_ERR(cublasDaxpy(cublas_handle, n, tmp, d_vtmp, 1, d_r, 1));
		CHECK_CUBLAS_ERR(cublasDdot(cublas_handle, n, d_r, 1, d_r, 1, d_rdot));

		// terminate if the length (squared) residue is small enough
		CHECK_CUDA_ERR(cudaMemcpy(scalar_r, d_rdot, elem_size, cudaMemcpyDeviceToHost));
		if (*scalar_r < params.epsilon)
			break;
	
		// calculate new beta
		scalar_div<<<1, 1>>>(d_rdot, d_rprevdot, d_beta);

		// calculate new p
		CHECK_CUDA_ERR(cudaMemcpy(tmp, d_beta, elem_size, cudaMemcpyDeviceToHost)); // hack alpha cant be on device memory space
		CHECK_CUBLAS_ERR(cublasDscal(cublas_handle, n, tmp, d_p, 1)); 
		CHECK_CUBLAS_ERR(cublasDaxpy(cublas_handle, n, &ONE, d_r, 1, d_p, 1));

		iteration++;
	}

	CHECK_CUBLAS_ERR(cublasGetVector(n, sizeof(double), d_x, 1, x, 1));

	CHECK_CUDA_ERR(cudaFree(d_A));
	CHECK_CUDA_ERR(cudaFree(d_b));
	CHECK_CUDA_ERR(cudaFree(d_x));
	CHECK_CUDA_ERR(cudaFree(d_r));
	CHECK_CUDA_ERR(cudaFree(d_vtmp));
}


