// =================================================================
//
// File: Version_1_Secuencial.cpp
// Project: Actividad 3 - Verificación de números primos
// Author: Fermín Nieto
// Description: Da el resultado de la suma de los primeros 5000000 de numeros. Forma secuencial.
//
// =================================================================

/*
    Resultado de la suma de los primos entre 1 y 5000000: 838596693108
    Tiempo de ejecución paralela: 266 ms
    Tiempo de ejecución secuencial: 1729 ms
    Numero de hilos utilizados: 11
    Speedup: 6.5
    Eficiencia: 0.59 (59%)
*/

#include <iostream>
#include <cmath>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

using namespace std;
using namespace std::chrono;

const int LIMITE = 5000000;
const unsigned int NUM_THREADS = thread::hardware_concurrency();

long long sumatoria = 0;
mutex mtx;

bool esPrimo(int n) {
    if (n < 2)
        return false;
    int limite = static_cast<int>(sqrt(n));
    for (int i = 2; i <= limite; ++i) {
        if (n % i == 0)
            return false;
    }
    return true;
}


void sumaPrimos(int id_hilo) {
    int bloque = LIMITE / NUM_THREADS;
    int inicio = id_hilo * bloque + 1;
    int fin = (id_hilo == NUM_THREADS - 1) ? LIMITE : (id_hilo + 1) * bloque;

    long long suma_local = 0;
    for (int i = inicio; i <= fin; ++i) {
        if (esPrimo(i)) {
            suma_local += i;
        }
    }

    lock_guard<mutex> lock(mtx);
    sumatoria += suma_local;
}

int main() {
    cout << "Usando " << NUM_THREADS << " hilos." << endl;

    vector<thread> hilos;

    auto inicio = high_resolution_clock::now();

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        hilos.emplace_back(sumaPrimos, i);
    }

    for (auto& hilo : hilos) {
        hilo.join();
    }

    auto fin = high_resolution_clock::now();
    auto duracion = duration_cast<milliseconds>(fin - inicio);

    cout << "El resultado de la suma de los primos entre 1 y " << LIMITE << " es: " << sumatoria << endl;
    cout << "Tiempo de ejecución paralela: " << duracion.count() << " ms" << endl;

    return 0;
}
