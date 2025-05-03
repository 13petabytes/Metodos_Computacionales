## Complejidad del código:

Se ha de recalcar que la complejidad de cada sector del código se remarca por separado en los comentarios del mismo, en esta sección se detallara la complejidad total y el porqué de la misma.

Para entender la complejidad del código tenemos que entender las partes que lo componen y el manejo de las variables. En el código se insertan dos cadenas profundas, una que da la información del autómata y la otra que inserta la entrada a evaluar. La principal función a tener en cuenta estaría en la parte del código:

<div align="center">
(map (lambda (c) (acepta? c)) cadenas))
</div>


Esta parte emplea dos otras funciones, de complejidad O(L x T), acepta?, y el número de entradas que se evalúan. Pudiéndose definir en primera instancia la complejidad del programa como:

<div align="center">
O(C * L * T)
</div>


Pero vamos a matizar lo anterior, ya que se podría concluir que a este análisis le falta la consideración de tres apartados del programa, la reparación de la primera entrada en sub variables, la función “tabla-transiciones” y la interacción con el usuario. Primero podemos descartar la división de la variable autómata, ya que cada una de dichas particiones posee una complejidad de O(1), resultando risible para la complejidad total. Segundo tendríamos la complejidad de la función “tabla-transiciones”, esta se encarga de de la construcción del mapa de posibles transiciones presente en el autómata dado,  poseyendo una complejidad de O(T), siendo T el número de transiciones, porque podríamos agregar a la ecuación final esta complejidad como una suma, al ser efectuada solo una vez:

<div align="center">
O(T) + O(C * T * L)
</div>


Si bien la ecuación anterior estaría correcto en un inicio, en términos asintóticos, al solo reprise una vez la ecuación, resulta innecesario el incluirlo en la complejidad general, al no ser un término de mayor orden, como lo sería la complejidad de la última sección del programa, lo mismo estaría ocurriendo con la interacción con el usuario, al esta también tener sólo una iteración.

Pudiéndose concluir que la complejidad del programa, en efecto es:


<div align="center">
O(C * L * T)
</div>


