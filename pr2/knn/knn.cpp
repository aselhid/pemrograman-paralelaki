#include <vector>
#include <iostream>
#include <queue>
#include <map>
#include <cmath>
#include <cassert>
#include <utility>
#include <omp.h>
#include <fstream>
#include <chrono>

using namespace std;

class Timer
{
public:
    Timer(string timer_name)
    {
        name = timer_name;
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
        cout << name << ": " << total.count() << endl;
    }

private:
    std::chrono::duration<double> total;
    std::chrono::time_point<std::chrono::high_resolution_clock> begin;
    string name;
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
    Timer total_time("total_time");
    total_time.start();

    input.open("in");
    input >> n >> t >> K;

    double *inputX = new double[n];
    double *inputY = new double[n];
    int *inputC = new int[n];

    double *testX = new double[t];
    double *testY = new double[t];
    int *testC = new int[t];

    for (int i = 0; i < n; i++)
    {
        input >> inputX[i] >> inputY[i] >> inputC[i];
    }

    for (int i = 0; i < t; i++)
    {
        input >> testX[i] >> testY[i] >> testC[i];
    }

    int correct = 0;

#pragma omp parallel for default(shared) reduction(+:correct)  
    for (int tc = 0; tc < t; tc++)
    {
        double x = inputX[tc], y = inputY[tc];
        int c = inputC[tc];

        priority_queue<pair<double, int>> distances;
        for (int i = 0; i < n; i++)
        {
            double dist = distance(inputX[i], x, inputY[i], y);
            distances.push(make_pair(dist, inputC[i]));
            
            while (distances.size() > K)
            {
                distances.pop();
            }
        }

        map<int, int> freq;
        while (!distances.empty())
        {
            pair<double, int> top = distances.top();
            freq[top.second]++;
            distances.pop();
        }

        auto highest = freq.end();
        --highest;

        int pred_class = highest->first;
        if (pred_class == c)
        {
            correct++;
        }
    }

    printf("Accuracy: %f (%d/%d)\n", 1. * correct / t, correct, t);

    total_time.stop();
    total_time.print();

    delete[] inputX;
    delete[] inputY;
    delete[] inputC;
    delete[] testX;
    delete[] testY;
    delete[] testC;
}