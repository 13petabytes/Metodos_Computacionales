#include <iostream>
#include <chrono>

#define N 10000000

using namespace std;
using namespace chrono;

int main() {
    auto start = high_resolution_clock::now();

    double pi = 0.0;
    for (int i = 0; i < N; i++) {
        pi += ((i % 2 == 0) ? 1.0 : -1.0) / (2 * i + 1);
    }

    pi *= 4;

    auto end = high_resolution_clock::now();
    double time = duration<double, milli>(end - start).count();

    cout << "Pi aproximado: " << pi << "\n";
    cout << "Tiempo secuencial: " << time << " ms\n";

    return 0;
}
