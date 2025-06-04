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
    Tiempo: 1729 ms
*/
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Algoritmo para determinar si un número n es primo
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

// Función que suma los números primos en un rango
long long sumaPrimos() {
    long long suma = 0;
    for (int i = 1; i <= 5000000; ++i) {
        if (esPrimo(i))
            suma += i;
    }
    return suma;
}

int main() {
    auto inicio = high_resolution_clock::now();

    long long sumatoria = sumaPrimos();

    auto fin = high_resolution_clock::now();
    auto duracion = duration_cast<milliseconds>(fin - inicio);

    cout << "El resultado de la suma de los primos entre 1 y 5000000 es: " << sumatoria << endl;
    cout << "Tiempo de ejecución secuencial: " << duracion.count() << " ms" << endl;

    return 0;
}
