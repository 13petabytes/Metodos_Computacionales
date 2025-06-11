# Reporte Actividad Integradora N.3

## Introducción

El proyecto actual, se presenta como un ejercicio de testeo de testeo de velocidad de compilado de un mismo ejercicio en diversos métodos de ejecución, dichos métodos son los siguientes, ranqueados de mayor tiempo de ejecución a menor: 


Secuencial, que indica que solo se ejecuta en un solo procesador o hilo. 

Paralelo, que se ejecuta en los números de hilos o procesadores determinado por el programa, 11 en este caso.

CUDA, que es una plataforma de programación paralela desarrollada por nvidia, que permite a los desarrolladores acelerar la ejecución de programas muy demandantes, en este caso se emplearon 512 bloques, y 32 hilos , generando así una división en forma de matriz, a diferencia de la programación en paralelo tradicional, limitada a 2 dimensiones.

En este caso, estos métodos serán sometidos a la tarea de sumar los números primos menores a 5 millones, como tal todos cumplen dicha tarea, dando como resultado 838596693108, pero de nuevo, lo importante no es esto, sino en cuanto tiempo lo hacen.

## Soluciones

### Desarrollo

Como tal todas las versiones realizadas siguiendo la misma estructura, main define el número de hilos en los que será dividida la tarea, de ser necesario, para posteriormente llamar a “sumarPrimos”, función que con la ayuda de “esPrimo”, para detectar los números requeridos, va sumando de uno en uno los números. La complejidad seria representada de la siguiente forma:

<br>
<div align="center">
O(n√n)
</div>
</br>

Esto se debe a que por cada número entre 1 y n se realiza una verificación de primalidad que, en el peor de los casos, recorre hasta su raíz cuadrada, generando así una complejidad compuesta que crece como el producto entre n y la raíz de n. Todos los programas comparten la misma complejidad.

En los casos en los que se empleó la división por hilos y la tecnología CUDA, se divide el segmento de número a checar y sumar, en el número correspondiente de hilos, para posteriormente sumar las sumas echas por cada hilo. En el caso de CUDA, esto se profundiza, al tener 512 bloques, de cada uno 32 hilos, se podría decir que la división es similar a una matriz, donde cada columna es un arreglo de hilos, profundizando aún más la división del trabajo al elevarla a una tercera dimensión, y similar a los apuntes de Adam Smith, se eficienta más el tiempo de trabajo a más división de trabajo exista. Por lo que al comparar los 11 hilos que están trabajando en el programa con programación paralela con los 16,384 del desarrollado con CUDA podemos presuponer de antemano, cuál será más rápido.

### Tiempo de ejecución y speed up

Procediendo a los resultados, estos son los tiempos de ejecución de cada programa:

![image](https://github.com/user-attachments/assets/c837db2b-b62d-4adc-a601-05f1d0e41017)


	
Con esto podemos calcular tanto la eficiencia que las versiones que dividen presentan sobre el modelo de ejecución secuencial, esto empleando el promedio del tiempo que tomo a cada una de las ejecuciones, permitiendo así el tener una vara de medir igual para cada método que busca mejorar el tiempo del algoritmo secuencial. Para esto se emplean las siguientes ecuaciones:


<br>
<div align="center">
	
Speed up = Tiempo de ejecución secuencial / Tiempo de ejecución a comparar

Eficiencia = Speed up / Número de procesadores totales empleados

</div>
</br>


Tabla con cálculos:

![image](https://github.com/user-attachments/assets/ddfdef4c-4551-49dd-81fe-2329e3c0a7f4)



### Conclusiones de los resultados

Al observar los resultados podemos confirmar los aseverado anteriormente, y es que el programa que emplea CUDA presentó una menor tiempo de ejecución que el resto, a al par de un mayor speed up. Pero algo curioso se presenta en el cálculo de la eficiencia, y que el programa que solo divide en 11 hilos presenta una tres dígitos mayor que la del desarrollado con cuidado. Pudiendo concluir así, que el tiempo que ahorra cada hilo agregado a la ejecución va bajando su valor conforma más hilos existan ejecutando dicha acción actualmente, o en otras palabras, el valor de división de trabajo que cada hilo presenta a un programa de n hilos es menor que si el programa tuviera n -1 hilos. 

Para mejor entendimiento de esto, tenemos esta gráfica, donde las abscisas representan el número de hilos y las abscisas la eficiencia, el punto representa el número de hilos y la eficiencia del programa con 11 hilos (11,6.5/11):

![image](https://github.com/user-attachments/assets/224ea768-a0b1-4826-98c2-25f66e4d29f5)


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

![image](https://github.com/user-attachments/assets/b91ae663-9abd-4fcb-af7c-bd2ffb8df7e1)


Al observar el comportamiento del speed up en función del número de hilos 
𝑃
P, notamos que la aceleración crece rápido al inicio, pero conforme aumentan los hilos, la mejora se vuelve cada vez más lenta. Por eso se eligió una función logarítmica para ajustarlo, que tiene la forma:

<br> 
<div align="center">
	S(P) = 5.0942 + 0.5862 * ln(P)
</div> 
</br>

Los números 5.0942 y 0.5862 son constantes calculadas a partir de los dos puntos conocidos: un speed up de 6.5 con 11 hilos y uno de 10.7832 con 16384 hilos. Para obtener estos valores se resolvió un sistema de ecuaciones usando el logaritmo natural de 
𝑃
P. Esta ecuación no pretende ser una fórmula exacta de eficiencia, sino una aproximación que refleja el comportamiento observado en los datos.

De esta forma, se puede demostrar que el speed up crece pero con un rendimiento decreciente conforme se agregan más hilos, concepto que también es consistente con la Ley de Amdahl.

Para entender cómo se obtuvieron los coeficientes, se resolvió el siguiente sistema con los puntos conocidos:

<br> 
<div align="center"> 
	6.5 = a + b * ln(11)
	
</div> 
<div align="center"> 
10.7832 = a + b * ln(16384)
</div>
</br>

De donde se obtuvo:

<br> <div align="center">

a = 5.0942

<div align="center"> 
b = 0.5862
</div>
</div> </br>
y así la función ajustada queda:

<br> <div align="center"> S(P) = 5.0942 + 0.5862 * ln(P)</div> </br>
Por otro lado, la Ley de Amdahl permite calcular la mejora máxima que se puede obtener al paralelizar una fracción 
𝑝
p de un sistema, según la ecuación:

<br> <div align="center">  S = 1  /  (1 - p + p / s) </div> </br>
Donde 
𝑝
p es la fracción paralelizable del sistema y 
𝑠
s el factor de aceleración de dicha parte. Eugene Myron Amdahl indica que, al tender 
𝑝
p a infinito (en este caso, 
𝑝
≈
0.907
p≈0.907), el valor resultante del speed up es cercano a 10.75, lo cual concuerda con el valor experimental obtenido para 16384 hilos. Esto confirma que el modelo logarítmico y la Ley de Amdahl se complementan en la explicación del comportamiento del speed up.

En conclusión, un mayor número de hilos reduce el tiempo de ejecución, pero el beneficio que aporta cada hilo disminuye conforme aumenta el total de hilos ejecutándose.

## Conclusiones personales del proyecto y la clase

Si bien no voy a continuar mi vida universitaria en la carrera de ITC, he podido disfrutar de la clase, y en particular de este proyecto,  debido a su relación con teoremas matemáticos, en los que me hubiera gustado profundizar más. Pude darme cuenta que genuinamente me faltaban matemáticas para darle cuerpo a la teoría con la que trabajaba. Me llevo buenas experiencias y necesarias reflexiones sobre mi deseo vocacional. Muchas gracias por el semestre.


# Referencias:

Amdahl, Gene M. (1967). "Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities" (PDF). AFIPS Conference Proceedings (30): 483–485. doi:10.1145/1465482.1465560. S2CID 195607370.


