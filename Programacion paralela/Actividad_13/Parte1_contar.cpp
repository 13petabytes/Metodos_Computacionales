// =================================================================
//
// File:    Parte1_contar.cpp
// Author: Fermín Nieto 
// Description: Contar el número de pares que existen en un arreglo de enteros. El tamaño del arreglo debe ser 1x109 (1,000,000,000).
//
// =================================================================

/*
Salida y calculo de eficiencia:
Número de pares: 499999851
Tiempo paralelo: 5199.57 ms
Tiempo secuencial: 21510 ms
Speedup: 4.13688
Nucleos: 11
Eficiencia: 0.37608
*/

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>

#define SIZE 1000000000
#define THREADS std::thread::hardware_concurrency()

using namespace std;
using namespace chrono;

void count_even(int *arr, long long start, long long end, long long &count) {
    count = 0;
    for (long long i = start; i < end; i++) {
        if (arr[i] % 2 == 0) count++;
    }
}

int main() {
    int *arr = new int[SIZE];
    //randomización de numeros
    for (int i = 0; i < SIZE; i++) {
        arr[i] = rand();  
    }

    thread threads[THREADS];
    long long counts[THREADS];

    auto start = high_resolution_clock::now();

    long long block_size = SIZE / THREADS;

    for (int i = 0; i < THREADS; i++) {
        long long s = i * block_size;
        long long e = (i == THREADS - 1) ? SIZE : s + block_size;
        threads[i] = thread(count_even, arr, s, e, ref(counts[i]));
    }

    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    long long total = 0;
    for (int i = 0; i < THREADS; i++) {
        total += counts[i];
    }

    auto end = high_resolution_clock::now();
    double time = duration<double, milli>(end - start).count();

    cout << "Número de pares: " << total << "\n";
    cout << "Tiempo paralelo: " << time << " ms\n";

    delete[] arr;
    return 0;
}
