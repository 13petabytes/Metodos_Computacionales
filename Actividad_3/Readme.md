# Reporte Actividad Integradora N.3

## Introducción

El proyecto actual, se presenta como un ejercicio de testeo de testeo de velocidad de compilado de un mismo ejercicio en diversos métodos de ejecución, dichos métodos son los siguientes, ranqueados de mayor tiempo de ejecución a menor: 


Secuencial, que indica que solo se ejecuta en un solo procesador o hilo. 

Paralelo, que se ejecuta en los números de hilos o procesadores determinado por el programa, 11 en este caso.

CUDA, que es una plataforma de programación paralela desarrollada por nvidia, que permite a los desarrolladores acelerar la ejecución de programas muy demandantes, en este caso se emplearon 512 bloques, y 32 hilos , generando así una división en forma de matriz, a diferencia de la programación en paralelo tradicional, limitada a 2 dimensiones.

En este caso, estos métodos serán sometidos a la tarea de sumar los números primos menores a 5 millones, como tal todos cumplen dicha tarea, dando como resultado 838596693108, pero de nuevo, lo importante no es esto, sino en cuanto tiempo lo hacen.

## Soluciones

### Desarrollo

Como tal todas las versiones realizadas siguiendo la misma estructura, main define el número de hilos en los que será dividida la tarea, de ser necesario, para posteriormente llamar a “sumarPrimos”, función que con la ayuda de “esPrimo”, para detectar los números requeridos, va sumando de uno en uno los números. En los casos en los que se empleó la división por hilos y la tecnología CUDA, se divide el segmento de número a checar y sumar en el número correspondiente de hilos, para posteriormente sumar cada resultado. En el caso de CUDA, esto se profundiza, al tener 512 bloques, de cada uno 32 hilos, se podría decir que la división es similar a una matriz, donde cada columna es un arreglo de hilos, profundizando aún más la división del trabajo al darle tercera dimensión, y similar a los apuntes de Adam Smith, se eficienta más el tiempo a más división de trabajo exista. Por lo que al comparar los 11 hilos que están trabajando en el programa con programación paralela con los 16,384 del desarrollado con CUDA, podemos presuponer de antemano, cuál será más rápido.

### Tiempo de ejecución y speed up

Procediendo a los resultados, estos son los tiempos de ejecución de cada programa:


	
Con esto podemos calcular tanto la eficiencia que las versiones que dividen presentan sobre el modelo de ejecución secuencial, permitiendo así, el tener una vara de medir igual para cada método que busca mejorar el tiempo del algoritmo secuencial. Para esto se emplean las siguientes ecuaciones:


<br>
<div align="center">
	
Speed up = Tiempo de ejecución secuencial / Tiempo de ejecución a comparar

Eficiencia = Speed up / Número de procesadores totales empleados

</div>
</br>


Tabla con cálculos:


### Conclusiones de los resultados

Al observar los resultados podemos confirmar los aseverado anteriormente, y es que el programa que emplea CUDA presentó una menor tiempo de ejecución que el resto, a al par de un mayor speed up. Pero algo curioso se presenta en el cálculo de la eficiencia, y que el programa que solo divide en 11 hilos presenta una tres dígitos mayor que la del desarrollado con cuidado. Pudiendo concluir así, que el tiempo que ahorra cada hilo agregado a la ejecución va bajando su valor conforma más hilos existan ejecutando dicha acción actualmente, o en otras palabras, el valor de división de trabajo que cada hilo presenta a un programa de n hilos es menor que si el programa tuviera n -1 hilos. 

Para mejor entendimiento de esto, tenemos esta gráfica, donde las abscisas representan el número de hilos y las abscisas la eficiencia, el punto representa el número de hilos y la eficiencia del programa con 11 hilos (11,6.5/11):


La ecuación empleada para esto es la siguiente:

<br>
<div align="center">
	
E(P) = S(P) / P

</div>
</br>
	
Donde E(P) representa el cuánto valdrá  la eficiencia, S(P) el Speed up en base al número de hilos y P el número de hilos. Como se puede concluir a simple vista, el valor que nos da un speed up también nos la da una función. Esta se define de la siguiente forma:

<br>
<div align="center">
S(P)= 5.0942 + 0.5862 * ln(P)
</div>
</br>

Gráfica de la fórmula. El eje x representa el número de hilos y el eje y el incremento en la eficiencia:



En este caso, los números presentes son las eficiencias obtenidas y el ln(P), el porqué se escogió una ecuación con estas propiedades, se debe al comportamiento que presentó el speed up, al ser abrupto al inicio. Como tal la ecuación no presenta una eficiencia infalible, pero sí una considerable, pudiendo demostrar esto con la ley de Amdahl.

<br>
<div align="center">
S = 1  /  (1 - p + p / s)
</div>
</br>

Este teorema permite calcular la mejora máxima de un sistema, donde p es la fracción del sistema, o para fines del proyecto, la fracción paralelizable del sistema,  y s el factor de aceleración de dicha parte. Eugene Myron Amdahl también indica que, al tender p a infinito (para el cálculo p ≈ 0.907), el valor resultante sería similar a 10.75, concordando con el valor obtenido en las pruebas, y en la ecuación al calcular con P igual a 16384, corresponde con el aproximado a infinito que el teorema propone.


## Conclusiones personales del proyecto y la clase

Si bien no voy a continuar mi vida universitaria en la carrera de ITC, he podido disfrutar de la clase, y en particular de este proyecto,  debido a su relación con teoremas matemáticos, en los que me hubiera gustado profundizar más. Pude darme cuenta que genuinamente me faltaban matemáticas para darle cuerpo a la teoría con la que trabajaba. Me llevo buenas experiencias y necesarias reflexiones sobre mi deseo vocacional. Muchas gracias por el semestre.


#Referencias:
Amdahl, Gene M. (1967). "Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities" (PDF). AFIPS Conference Proceedings (30): 483–485. doi:10.1145/1465482.1465560. S2CID 195607370.


