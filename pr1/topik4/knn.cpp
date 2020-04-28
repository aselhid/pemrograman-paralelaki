#include <vector>
#include <iostream>
#include <queue>
#include <map>
#include <cmath>
#include <cassert>
#include <utility>
#include <mpi.h>
#include <fstream>
#include <chrono>

using namespace std;

class Timer
{
public:
    Timer(string timer_name, int _rank)
    {
        name = timer_name;
        rank = _rank;
    }
    void start()
    {
        begin = std::chrono::high_resolution_clock::now();
    }
    void stop()
    {
        total += std::chrono::high_resolution_clock::now() - begin;
    }
    void print()
    {
        cout << name << '(' << rank << "): " << total.count() << endl;
    }

private:
    std::chrono::duration<double> total;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin;
    string name;
    int rank;
};

double distance(double x1, double x2, double y1, double y2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

int K = 77;
int n, t;

ifstream input;

int main(int argc, char *argv[])
{
    int rank, numprocs, root = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    Timer total_time("total_time", rank), comm_time("comm_time", rank);
    total_time.start();

    if (rank == 0)
    {
        input.open("in");

        input >> n >> t >> K;
        assert(n % numprocs == 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    comm_time.start();
    MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    comm_time.stop();

    int tasks = n / numprocs;

    double *inputX = new double[n];
    double *inputY = new double[n];
    int *inputC = new int[n];

    double *trainX = new double[tasks];
    double *trainY = new double[tasks];
    int *trainC = new int[tasks];

    double *testX = new double[t];
    double *testY = new double[t];
    int *testC = new int[t];

    double *dists = new double[numprocs * K];
    int *classes = new int[numprocs * K];

    double *local_dists = new double[K];
    int *local_classes = new int[K];

    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
        {
            input >> inputX[i] >> inputY[i] >> inputC[i];
        }

        for (int i = 0; i < t; i++)
        {
            input >> testX[i] >> testY[i] >> testC[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    comm_time.start();
    MPI_Scatter(inputX, tasks, MPI_DOUBLE, trainX, tasks, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(inputY, tasks, MPI_DOUBLE, trainY, tasks, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(inputC, tasks, MPI_INT, trainC, tasks, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    comm_time.stop();

    int correct = 0;
    for (int tc = 0; tc < t; tc++)
    {
        double x, y;
        int c;

        if (rank == 0)
        {
            x = inputX[tc];
            y = inputY[tc];
            c = inputC[tc];
        }

        MPI_Barrier(MPI_COMM_WORLD);
        comm_time.start();
        MPI_Bcast(&x, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&y, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        comm_time.stop();

        priority_queue<pair<double, int>> distances;
        for (int i = 0; i < tasks; i++)
        {

            double dist = distance(trainX[i], x, trainY[i], y);
            distances.push(make_pair(dist, trainC[i]));
            while (distances.size() > K)
            {
                distances.pop();
            }
        }

        int idx = 0;
        while (!distances.empty())
        {
            local_dists[idx] = distances.top().first;
            local_classes[idx] = distances.top().second;
            idx++;
            distances.pop();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        comm_time.start();
        MPI_Gather(local_dists, K, MPI_DOUBLE, dists, K, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Gather(local_classes, K, MPI_INT, classes, K, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        comm_time.stop();

        // merge disini
        if (rank == 0)
        {
            priority_queue<pair<double, int>> pq;

            int cnt = 0;
            for (int i = 0; i < numprocs * K; i++)
            {
                pq.push(make_pair(dists[i], classes[i]));
                while (pq.size() > K)
                {
                    pq.pop();
                }
            }

            map<int, int> freq;
            while (!pq.empty())
            {
                pair<double, int> top = pq.top();
                freq[top.second]++;
                pq.pop();
            }

            auto highest = freq.end();
            --highest;

            int pred_class = highest->first;

            if (pred_class == c)
            {
                correct++;
            }
        }
    }

    if (rank == 0)
    {
        printf("Accuracy: %f (%d/%d)\n", 1. * correct / t, correct, t);
    }
    total_time.stop();
    total_time.print();
    comm_time.print();

    MPI_Finalize();

    delete[] inputX;
    delete[] inputY;
    delete[] inputC;
    delete[] trainX;
    delete[] trainY;
    delete[] trainC;
    delete[] testX;
    delete[] testY;
    delete[] testC;
    delete[] dists;
    delete[] classes;
    delete[] local_dists;
    delete[] local_classes;
}
