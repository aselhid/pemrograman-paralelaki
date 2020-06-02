#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CHECK_CUDA_ERR(err) { int e = (err); if (cudaSuccess != e) { printf("CUDA error No. %d in %s at line %d\n", e, __FILE__, __LINE__); exit(EXIT_FAILURE); } }

#define CHECK_CUBLAS_ERR(err) { int e = (err); if (CUBLAS_STATUS_SUCCESS != e) { printf("CUBLAS error No. %d in %s at line %d\n", e, __FILE__, __LINE__); exit(EXIT_FAILURE); } }

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

const double ZERO = 0.0;
const double ONE = 1.0;
const double NEG_ONE = -1.0;

typedef struct {
	double epsilon;
	unsigned int max_iteration;
} conjugrad_params;

void gpu_conjugate_gradient(double* A, double* b, int n, conjugrad_params params, double* x);
