// =================================================================
// File: Ejercicio2_Act_14_PI.cu
// Author: ChatGPT adaptado para Fermín Nieto
// Description: Aproximación de Pi usando la serie de Gregory-Leibniz
//              con 1,000,000,000 términos, paralelo con CUDA.
// =================================================================

/*
    Pi aproximado (CUDA): 3.141592652589800
    Tiempo paralelo (CUDA): 21.862204000 ms
    Tiempo secuencial: 23.1508 ms
    Speedup: 1.0586
    BLOCKS = 512
    THREADS = 2048  
    Eficiencia: 1.0099×10^−6 (0.00010099%)
*/

/*
    Codigo para ejecutar en google colab:

    # Verificar GPU disponible
    !nvidia-smi

    # Verificar versión de nvcc (normalmente ya está instalado en Colab)
    !nvcc --version

    # Compilar el archivo CUDA con la arquitectura correcta para Tesla T4 (sm_75)
    !nvcc -arch=sm_75 -o Ejercicio2_Act_14_PI_exec Ejercicio2_Act_14_PI.cu

    # Ejecutar el programa compilado
    !./Ejercicio2_Act_14_PI_exec

*/

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

#define SIZE 1000000000ULL       // 1,000,000,000 términos
#define THREADS 512
#define BLOCKS 2048              

using namespace std;
using namespace chrono;

__global__ void pi_kernel(float *results, unsigned long long size) {
    __shared__ float cache[THREADS];

    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;
    float sum = 0.0f;

    // Cada hilo procesa múltiples términos con stride total
    while (tid < size) {
        float sign = (tid % 2 == 0) ? 1.0f : -1.0f;
        sum += sign / (2.0f * tid + 1.0f);
        tid += blockDim.x * gridDim.x;
    }

    // Guardar resultado parcial en memoria compartida
    cache[cacheIdx] = sum;
    __syncthreads();

    // Reducción en bloque para sumar resultados parciales
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
    }

    // El primer hilo de cada bloque escribe el resultado parcial
    if (cacheIdx == 0) {
        results[blockIdx.x] = cache[0];
    }
}

int main() {
    float *d_results;
    float *h_results = new float[BLOCKS];

    // Reservar memoria en device
    cudaMalloc((void**)&d_results, BLOCKS * sizeof(float));

    // Medir tiempo de ejecución paralela
    auto start = high_resolution_clock::now();

    pi_kernel<<<BLOCKS, THREADS>>>(d_results, SIZE);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();

    // Copiar resultados parciales al host
    cudaMemcpy(h_results, d_results, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

    // Sumar resultados parciales para obtener la aproximación final
    float pi_sum = 0.0f;
    for (int i = 0; i < BLOCKS; i++) {
        pi_sum += h_results[i];
    }

    float pi = pi_sum * 4.0f;

    double time = duration<double, milli>(end - start).count();

    cout << fixed << setprecision(9);
    cout << "Pi aproximado (CUDA): " << pi << "\n";
    cout << "Tiempo paralelo (CUDA): " << time << " ms\n";

    // Liberar memoria
    cudaFree(d_results);
    delete[] h_results;

    return 0;
}

