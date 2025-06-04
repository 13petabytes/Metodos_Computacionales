#include <iostream>
#include <chrono>
#include <cstdlib>

#define SIZE 1000000000

using namespace std;
using namespace chrono;

int main() {
    int *arr = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        arr[i] = rand();
    }

    auto start = high_resolution_clock::now();

    long long total = 0;
    for (long long i = 0; i < SIZE; i++) {
        if (arr[i] % 2 == 0) total++;
    }

    auto end = high_resolution_clock::now();
    double time = duration<double, milli>(end - start).count();

    cout << "Número de pares: " << total << "\n";
    cout << "Tiempo secuencial: " << time << " ms\n";

    delete[] arr;
    return 0;
}
