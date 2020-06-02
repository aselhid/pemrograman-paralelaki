#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "conjugrad_kernel.cuh"
#include "utils.h"

void generate_random_cublas_sym_positive_definite_matrix(double *M, int n, int multiplier)
{
	double *M_transpose;
	size_t matrix_size = sizeof(*M) * n * n;

	M_transpose = (double *)malloc(matrix_size);

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			double e = (double)rand() / RAND_MAX;
			M[IDX2C(j, i, n)] = e;
			M_transpose[IDX2C(i, j, n)] = e;
		}
	}

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			M[IDX2C(j, i, n)] = 0.5 * (M[IDX2C(j, i, n)] + M_transpose[IDX2C(j, i, n)]);

			if (i == j)
				M[IDX2C(j, i, n)] += n;

			M[IDX2C(j, i, n)] *= multiplier;
		}
	}
}

void generate_random_vector(double *v, int n, int multiplier)
{
	for (int i = 0; i < n; ++i)
		v[i] = (double)rand() / RAND_MAX * multiplier;
}

void matvec_multiplication(double *A, int n, int m, double *x, double *b)
{
	for (int i = 0; i < n; ++i)
	{
		b[i] = 0;
		for (int j = 0; j < m; ++j)
		{
			b[i] += A[IDX2C(j, i, m)] * x[j];
		}
	}
}

bool isAlmostEqual(double a, double b, double epsilon)
{
	return fabs(a - b) < epsilon;
}

int main()
{
	double *A, *x, *b, *x_parallel;
	double epsilon = 1e-6;
	int n = 1e4;

	A = (double *)malloc(sizeof(*A) * n * n);
	x = (double *)malloc(sizeof(*x) * n);
	x_parallel = (double *)malloc(sizeof(*x_parallel) * n);
	b = (double *)malloc(sizeof(*b) * n);

	generate_random_cublas_sym_positive_definite_matrix(A, n, 10);
	generate_random_vector(x, n, 10);

	matvec_multiplication(A, n, n, x, b);

	// for (int i = 0; i < n; ++i)
	// {
	// 	for (int j = 0; j < n; ++j)
	// 	{
	// 		printf("%.17g ", A[IDX2C(j, i, n)]);
	// 	}
	// 	printf("\n");
	// }
	// printf("\n");

	// for (int i = 0; i < n; ++i)
	// 	printf("%.17g ", x[i]);
	// printf("\n\n");

	// for (int i = 0; i < n; ++i)
	// 	printf("%.17g ", b[i]);
	// printf("\n\n");

	conjugrad_params params = {1e-10, (int)1e5};
	auto start = Clock::now();
	gpu_conjugate_gradient(A, b, n, params, x_parallel);
	auto finish = Clock::now();

	long long int d = (int)((finish - start).count() / 1e6);
	printf("%lld\n", d);

	// for (int i = 0; i < n; ++i)
	// 	printf("%.17g ", x_parallel[i]);

	bool isEqual = true;
	for (int i = 0; i < n; ++i)
	{
		if (!isAlmostEqual(x[i], x_parallel[i], epsilon))
		{
			isEqual = false;
			break;
		}
	}
	printf(isEqual ? "EQUAL\n" : "NOT\n");
}