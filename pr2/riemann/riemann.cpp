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

double parallel_calculate_y(double coefs[], int length, double x)
{
    double result = 0.0;
#pragma omp parallel for reduction(+ \
                                   : result)
    for (int i = 0; i < length; ++i)
    {
        result += pow(x, i) * coefs[i];
    }
    return result;
}

double parallel_riemann(double coefs[], int coefs_length, double a, double b, int n)
{
    double subinterval = (b - a) / n;
    double result = 0.0;
#pragma omp parallel for reduction(+ \
                                   : result)
    for (int i = 0; i < n; ++i)
    {
        double x = subinterval * i;
        result += subinterval * parallel_calculate_y(coefs, coefs_length, x);
    }
    return result;
}

double calculate_y(double coefs[], int length, double x)
{
    double result = 0.0;
    for (int i = 0; i < length; ++i)
    {
        result += pow(x, i) * coefs[i];
    }
    return result;
}

double riemann(double coefs[], int coefs_length, double a, double b, int n)
{
    double subinterval = (b - a) / n;
    double result = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double x = subinterval * i;
        result += subinterval * calculate_y(coefs, coefs_length, x);
    }
    return result;
}

int main()
{
    int MAX_POLYNOMIAL_DEGREE = 1000000;
    double coefs[MAX_POLYNOMIAL_DEGREE];

    for (int i = 0; i < MAX_POLYNOMIAL_DEGREE; ++i)
    {
        coefs[i] = rand();
    }

    int partition = 100;

    printf("%d\n", omp_get_max_threads());

    timestamp_t before_sequential = get_timestamp();
    double sequential_result = riemann(coefs, MAX_POLYNOMIAL_DEGREE, 1, 2, partition);
    timestamp_t after_sequential = get_timestamp();

    timestamp_t before_parallel = get_timestamp();
    double parallel_result = parallel_riemann(coefs, MAX_POLYNOMIAL_DEGREE, 1, 2, partition);
    timestamp_t after_parallel = get_timestamp();

    printf("sequential: %llu ms\n", after_sequential - before_sequential);
    printf("parallel: %llu ms\n", after_parallel - before_parallel);
    printf(sequential_result == parallel_result ? "EQUAL\n" : "NOT\n");

    return 0;
}