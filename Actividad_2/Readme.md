# Reporte Actividar integradora N.2

## Estructuración del código:
  El código presentado para esta entrega se compone de dos apartados principales: la función principal, llamada validate, y la interacción con el usuario. Esta última es bastante sencilla y no representa un elemento diferenciador del programa, por lo que no se profundizará en ella.

  Centrándonos en la función validate, esta se divide conceptualmente en cinco secciones.

  La primera, titulada “División de la variable autómata”, se encarga de descomponer la lista profunda proporcionada por el usuario en sus partes constitutivas: los estados, el alfabeto, las transiciones, el estado inicial y los estados finales. Esta separación facilita el manejo interno del autómata.

  En segundo lugar, se define la estructura “tabla-transiciones”, la cual convierte la lista de transiciones en una tabla de asociación. Esta tabla se utiliza posteriormente para consultar de manera eficiente si existe una transición válida desde un estado dado con un símbolo específico.

  La tercera parte incluye la definición de la función siguiente-estado, la cual consulta la tabla de transiciones y determina cuál es el estado siguiente a partir del estado actual y el símbolo leído.

  La cuarta sección está compuesta por la función acepta?, que se encarga de procesar una cadena de entrada paso a paso. Esta función usa una función recursiva interna llamada recorrer, que va avanzando por los símbolos de la cadena y aplicando las transiciones hasta llegar al final de la entrada. Si al terminar se encuentra en un estado de aceptación, la función devuelve #t; de lo contrario, devuelve #f.

  Finalmente, en la quinta parte se utiliza un map para aplicar la función acepta? a cada una de las cadenas de entrada proporcionadas, devolviendo una lista con los resultados respectivos.



## Complejidad del código:

  Se ha de recalcar que la complejidad de cada sector del código se remarca por separado en los comentarios del mismo, en esta sección se detallara la complejidad total y el porqué de la misma.

  Para entender la complejidad del código tenemos que entender las partes que lo componen y el manejo de las variables. En el código se insertan dos cadenas profundas, una que da la información del autómata y la otra que inserta la entrada a evaluar. La principal función a tener en cuenta estaría en la parte del código:

<br>
<div align="center">
(map (lambda (c) (acepta? c)) cadenas))
</div>
<br>

  Esta parte emplea dos otras funciones, de complejidad O(L x T), acepta?, y el número de entradas que se evalúan. Pudiéndose definir en primera instancia la complejidad del programa como:

<br>
<div align="center">
O(C * L * T)
</div>
<br>

  Pero vamos a matizar lo anterior, ya que se podría concluir que a este análisis le falta la consideración de tres apartados del programa, la reparación de la primera entrada en sub variables, la función “tabla-transiciones” y la interacción con el usuario. Primero podemos descartar la división de la variable autómata, ya que cada una de dichas particiones posee una complejidad de O(1), resultando risible para la complejidad total. Segundo tendríamos la complejidad de la función “tabla-transiciones”, esta se encarga de de la construcción del mapa de posibles transiciones presente en el autómata dado,  poseyendo una complejidad de O(T), siendo T el número de transiciones, porque podríamos agregar a la ecuación final esta complejidad como una suma, al ser efectuada solo una vez:

<br>
<div align="center">
O(T) + O(C * T * L)
</div>
<br>

  Si bien la ecuación anterior estaría correcto en un inicio, en términos asintóticos, al solo reprise una vez la ecuación, resulta innecesario el incluirlo en la complejidad general, al no ser un término de mayor orden, como lo sería la complejidad de la última sección del programa, lo mismo estaría ocurriendo con la interacción con el usuario, al esta también tener sólo una iteración.

  Pudiéndose concluir que la complejidad del programa, en efecto es:

<br>
<div align="center">
O(C * L * T)
</div>
<br>

## Reflección sobre la implementación:

Si bien el código presentado funciona correctamente, este mismo está sujeto a mejoras, ya que cabe la posibilidad de bajar su complejidad al cambiar la búsqueda lineal de siguiente-estado baje a O(1), esto con el uso de hash-table. Si bien se conoce la posibilidad de esto, al tener problemas en su implementación se descartó este método y se acabó optando por la aplicación de complejidad lineal.

## Reporte y reflexión final
La realización de este proyecto me permitió repasar y profundizar mis conocimientos en el lenguaje Racket, llevándome a explorar aspectos del lenguaje que van más allá de lo visto en clase. Esto fue necesario para comprender con mayor claridad cómo se manejan las variables internamente y cómo estructurar un código más eficiente y legible.

Asimismo, el proyecto representó un reto estimulante, no solo por el uso del lenguaje en sí, sino también por la necesidad de retomar y aplicar conceptos fundamentales de autómatas. Para ello, fue necesario revisar el proyecto anterior, lo que me permitió reforzar mi comprensión sobre la representación y manejo de autómatas dentro del código.

Si bien considero que la aplicación aún tiene áreas de oportunidad y puede ser optimizada, el resultado obtenido no me resulta insatisfactorio. Al contrario, me deja una base sólida sobre la cual seguir mejorando.

En conclusión, este proyecto fue una experiencia amena y retadora, que me ayudó a consolidar conocimientos tanto de programación funcional como de teoría de autómatas. Me motivó a ir más allá de lo teórico, aplicando conceptos de forma práctica en el desarrollo de una herramienta funcional.


