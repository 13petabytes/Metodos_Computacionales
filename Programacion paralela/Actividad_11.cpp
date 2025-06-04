// =================================================================
//
// File: Actividad_11.cpp
// Author: Fermín Nieto 
// Description: Crea 4 hilos, solo 2 pueden ejecutarse al mismo tiempo.
//              Cada hilo imprime "Ya me ejecuté" y termina.
//
// =================================================================

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

mutex mtx;
condition_variable cv;
int active_threads = 0;
const int MAX_ACTIVE_THREADS = 2;

void tarea(int id) {
    {
        // Bloquear hasta que haya espacio para ejecutar
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, []() { return active_threads < MAX_ACTIVE_THREADS; });
        active_threads++;
    }

    // Simular trabajo
    cout << "Hilo " << id << ": Ya me ejecuté" << endl;

    // Notificar que este hilo ha terminado
    {
        unique_lock<mutex> lock(mtx);
        active_threads--;
    }
    cv.notify_all(); // Avisar a los hilos en espera
}

int main() {
    thread threads[4];

    for (int i = 0; i < 4; i++) {
        threads[i] = thread(tarea, i + 1);
    }

    for (int i = 0; i < 4; i++) {
        threads[i].join();
    }

    return 0;
}
