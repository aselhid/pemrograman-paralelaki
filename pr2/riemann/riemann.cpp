#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>

typedef unsigned long long timestamp_t;

static timestamp_t get_timestamp()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

int is_double_almost_equal(double a, double b)
{
    return fabs(a - b) < .0001;
}

double riemann(double coefs[], int coefs_length, double a, double b, int n)
{
    double subinterval = (b - a) / n;
    double result = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double x = subinterval * i;
        double y = 0.0;

        for (int j = 0; j < coefs_length; ++j)
        {
            y += pow(x, j) * coefs[j];
        }

        result += subinterval * y;
    }

    return result;
}

double parallel_riemann(double coefs[], int coefs_length, double a, double b, int n)
{
    double subinterval = (b - a) / n;
    double result = 0.0;

#pragma omp parallel for collapse(2) reduction(+ \
                                               : result)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < coefs_length; ++j)
        {
            double x = subinterval * i;
            result += subinterval * (coefs[j] * pow(x, j));
        }
    }

    return result;
}

int main()
{
    int MAX_POLYNOMIAL_DEGREE = 99999 + 1;
    double coefs[MAX_POLYNOMIAL_DEGREE];
    int partition = 1000;

    for (int i = 0; i < MAX_POLYNOMIAL_DEGREE; ++i)
    {
        coefs[i] = rand();
    }

    riemann(coefs, MAX_POLYNOMIAL_DEGREE, 1, 2, partition);

    timestamp_t before_sequential = get_timestamp();
    double sequential_result = riemann(coefs, MAX_POLYNOMIAL_DEGREE, 1, 2, partition);
    timestamp_t after_sequential = get_timestamp();

    timestamp_t before_parallel = get_timestamp();
    double parallel_result = parallel_riemann(coefs, MAX_POLYNOMIAL_DEGREE, 1, 2, partition);
    timestamp_t after_parallel = get_timestamp();

    printf("1,%llu\n", after_sequential - before_sequential);
    printf("%d,%llu\n", omp_get_max_threads(), after_parallel - before_parallel);
    // printf(is_double_almost_equal(sequential_result, parallel_result) ? "EQUAL\n" : "NOT\n");

    return 0;
}