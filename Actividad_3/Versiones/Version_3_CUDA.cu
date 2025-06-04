// =================================================================
//
// File: Version_3_CUDA.cu
// Project: Actividad 3 - Verificación de números primos (CUDA)
// Author: Fermín Nieto
// Description: Da el resultado de la suma de los primeros 5000000 números primos.
//              Utiliza procesamiento paralelo con CUDA.
//
// =================================================================

/*
    Resultado de la suma de los primos entre 1 y 5000000: 838596693108
    Tiempo de ejecución CUDA: 160.342 ms
    Tiempo de ejecución secuencial: 1729 ms
    Speedup: 10.7832
    BLOCKS = 512;
    THREADS = 32;
    Eficiencia: 0.000658 (0.0658%)
*/

/*
    Codigo para compilar en google colab:

    !nvcc -arch=sm_75 -o Version_3_CUDA_exec Version_3_CUDA.cu
    !./Version_3_CUDA_exec

*/

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__device__ bool esPrimo(int n) {
    if (n < 2) return false;
    int raiz = (int)sqrtf((float)n);
    for (int i = 2; i <= raiz; ++i) {
        if (n % i == 0) return false;
    }
    return true;
}

__global__ void sumaPrimos(long long* resultados) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    long long suma_local = 0;

    // Repartir el trabajo entre los hilos
    for (int i = idx + 1; i <= 5000000; i += totalThreads) {
        if (esPrimo(i)) {
            suma_local += i;
        }
    }

    resultados[idx] = suma_local;
}

int main() {
    const int THREADS = 32;
    const int BLOCKS = 512;
    const int TOTAL_THREADS = THREADS * BLOCKS;

    long long* h_resultados = new long long[TOTAL_THREADS];
    long long* d_resultados;

    cudaMalloc(&d_resultados, TOTAL_THREADS * sizeof(long long));

    // Medir el tiempo de ejecución
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lanzar kernel con 32 x 512 configuración
    sumaPrimos<<<BLOCKS, THREADS>>>(d_resultados);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_resultados, d_resultados, TOTAL_THREADS * sizeof(long long), cudaMemcpyDeviceToHost);

    long long suma_total = 0;
    for (int i = 0; i < TOTAL_THREADS; i++) {
        suma_total += h_resultados[i];
    }

    std::cout << "Suma total de primos hasta 5000000: " << suma_total << std::endl;
    std::cout << "Tiempo de ejecución CUDA: " << ms << " ms" << std::endl;

    // Liberar memoria
    delete[] h_resultados;
    cudaFree(d_resultados);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
