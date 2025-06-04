// =================================================================
//
// File:  Parte2_PI.cpp
// Author: Fermín Nieto 
// Description: Calcular la aproximación de Pi usando la serie de Gregory-Leibniz. El valor de n debe ser 1x107 (10,000,000).
//
// =================================================================

/*
Salida y calculo de eficiencia:
Pi aproximado: 3.141591
Tiempo paralelo: 18.9531 ms
Tiempo secuencial: 23.1508 ms
Speedup: 1.22147
Nucleos: 11
Eficiencia: 0.11104
*/

#include <iostream>
#include <thread>
#include <chrono>

#define N 10000000
#define THREADS std::thread::hardware_concurrency()

using namespace std;
using namespace chrono;

void partial_pi(int start, int end, double &result) {
    result = 0.0;
    for (int i = start; i < end; i++) {
        result += ((i % 2 == 0) ? 1.0 : -1.0) / (2 * i + 1);
    }
}

int main() {
    thread threads[THREADS];
    double results[THREADS];

    auto start = high_resolution_clock::now();

    int block_size = N / THREADS;

    for (int i = 0; i < THREADS; i++) {
        int s = i * block_size;
        int e = (i == THREADS - 1) ? N : s + block_size;
        threads[i] = thread(partial_pi, s, e, ref(results[i]));
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    double pi = 0.0;
    for (int i = 0; i < THREADS; i++) {
        pi += results[i];
    }

    pi *= 4;

    auto end = high_resolution_clock::now();
    double time = duration<double, milli>(end - start).count();

    cout << "Pi aproximado: " << pi << "\n";
    cout << "Tiempo paralelo: " << time << " ms\n";

    return 0;
}

