// =================================================================
//
// File: Actividad_10.cpp
// Author: [Fermín Nieto]
// Description: Este programa crea dos hilos. Uno calcula raíces
//              cuadradas y otro calcula cuadrados. El segundo hilo
//              solo se ejecuta después de que el primero termina.
//
// =================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <thread>

using namespace std;

void raiz_cuadrada() {
    cout << "Raíces cuadradas:\n";
    for (int i = 1; i <= 10; i++) {
        cout << "√" << i << " = " << fixed << setprecision(3) << sqrt(i) << endl;
    }
}

void cuadrado() {
    cout << "Cuadrados:\n";
    for (int i = 1; i <= 10; i++) {
        cout << i << "^2 = " << (i * i) << endl;
    }
}

int main(int argc, char* argv[]) {
    bool x = true;
    while (x == true) {
        thread t1(raiz_cuadrada); // Primer hilo
        t1.join(); // Esperamos que termine
    
        thread t2(cuadrado); // Segundo hilo
        t2.join(); // Esperamos que termine también
    
    }

    return 0;
}
