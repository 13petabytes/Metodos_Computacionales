// =================================================================
//
// File: Ejercicio1_Act_14_contar.cu
// Author: Fermín Nieto
// Description: Cuenta el número de valores pares en un arreglo 
//              de enteros utilizando CUDA.
//
// =================================================================

/*
    Número de pares (CUDA): 500015853
    Tiempo paralelo (CUDA): 20.399 ms
    Tiempo secuencial: 21510 ms
    Speedup: 1054.04
    BLOCKS = 512
    THREADS = 2048
    Eficiencia: 0.001005 (0.1005%)
*/

/*
    Codigo para ejecutar en google colab:
    
    # Verificar GPU disponible
    !nvidia-smi

    # Verificar versión de nvcc (normalmente ya está instalado en Colab)
    !nvcc --version

    # Compilar el archivo CUDA con la arquitectura correcta para Tesla T4 (sm_75)
    !nvcc -arch=sm_75 -o Ejercicio1_Act_14_contar_exec Ejercicio1_Act_14_contar.cu

    # Ejecutar el programa compilado
    !./Ejercicio1_Act_14_contar_exec


*/

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#define SIZE 1000000000
#define THREADS 512
#define BLOCKS 2048 

using namespace std;
using namespace chrono;

// Kernel para contar pares
__global__ void count_even_kernel(int *arr, long long *results, int size) {
    __shared__ long long cache[THREADS];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    long long count = 0;

    while (tid < size) {
        if (arr[tid] % 2 == 0) {
            count++;
        }
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = count;
    __syncthreads();

    // Reducción en el bloque
    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Solo el primer hilo del bloque escribe el resultado parcial
    if (cacheIdx == 0) {
        results[blockIdx.x] = cache[0];
    }
}

int main() {
    int *h_array, *d_array;
    long long *h_results, *d_results;

    // Reservar memoria
    h_array = new int[SIZE];
    h_results = new long long[BLOCKS];

    // Inicializar con valores aleatorios
    for (int i = 0; i < SIZE; i++) {
        h_array[i] = rand();
    }

    // Reservar memoria en GPU
    cudaMalloc((void **) &d_array, SIZE * sizeof(int));
    cudaMalloc((void **) &d_results, BLOCKS * sizeof(long long));

    // Copiar arreglo a la GPU
    cudaMemcpy(d_array, h_array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Medir tiempo de ejecución
    auto start = high_resolution_clock::now();

    // Ejecutar kernel
    count_even_kernel<<<BLOCKS, THREADS>>>(d_array, d_results, SIZE);
    cudaDeviceSynchronize();  // Asegura que el kernel haya terminado

    // Copiar resultados parciales al host
    cudaMemcpy(h_results, d_results, BLOCKS * sizeof(long long), cudaMemcpyDeviceToHost);

    // Sumar resultados parciales
    long long total_even = 0;
    for (int i = 0; i < BLOCKS; i++) {
        total_even += h_results[i];
    }

    auto end = high_resolution_clock::now();
    double time = duration<double, milli>(end - start).count();

    // Mostrar resultados
    cout << "Número de pares (CUDA): " << total_even << "\n";
    cout << "Tiempo paralelo (CUDA): " << fixed << setprecision(3) << time << " ms\n";

    // Liberar memoria
    cudaFree(d_array);
    cudaFree(d_results);
    delete[] h_array;
    delete[] h_results;

    return 0;
}