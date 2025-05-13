// =================================================================
//
// File: Actividad_12.cpp
// Author: Fermín Nieto (modificado por ChatGPT)
// Description: Problema de producto-consumidor modificado. 
//
// =================================================================

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>

std::mutex mutex;
std::condition_variable cond_var;
int rebanadas = 0;
bool pizza_pedida = false;

void estudiante(int id) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex);

        while (rebanadas == 0) {
            if (!pizza_pedida) {
                pizza_pedida = true;
                std::cout << "Estudiante " << id << " llama a la pizzería.\n";
                cond_var.notify_all(); // Notifica a la pizzería
            }
            std::cout << "Estudiante " << id << " duerme esperando pizza...\n";
            cond_var.wait(lock); // Espera hasta que haya pizza
        }

        // Comer una rebanada
        rebanadas--;
        std::cout << "Estudiante " << id << " toma una rebanada. Rebanadas restantes: " << rebanadas << "\n";

        lock.unlock();

        // Simula codificación mientras come pizza
        std::this_thread::sleep_for(std::chrono::milliseconds(100 + rand() % 200));
    }
}

void pizzeria() {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        while (!pizza_pedida) {
            cond_var.wait(lock); // Espera hasta que alguien pida pizza
        }

        // "Entrega" la pizza
        std::cout << "📦 La pizzería entrega una nueva pizza\n";
        rebanadas = 8;
        pizza_pedida = false;

        lock.unlock();
        cond_var.notify_all(); // Despierta a los estudiantes
    }
}

int main() {
    std::thread hilo_pizzeria(pizzeria);
    std::vector<std::thread> hilos_estudiantes;

    for (int i = 0; i < 5; ++i) {
        hilos_estudiantes.emplace_back(estudiante, i + 1);
    }

    hilo_pizzeria.join();
    for (auto& hilo : hilos_estudiantes) {
        hilo.join();
    }

    return 0;
}
