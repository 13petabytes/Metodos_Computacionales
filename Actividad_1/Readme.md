# Reporte Actividad Integradora N.1

## Reflexión y reporte:


El proyecto está separado en tres partes importantes, la primera está conformada por la función, que se encuentra en el main.cpp, “string shuntingYard(const string& regex)”. Esta función tiene la responsabilidad de transformar una expresión regular en una forma más legible para el objeto NFA, en la forma de un postfijo. Para esto nos hacemos uso del Algoritmo shunting yard, el cual se piensa para transformar en infijo a postfijo. Como ejemplo, la expresión (a|b)*abb, se convierte en ab|*.a.b.b.

Se tiene que clarificar que en el caso de expresarse la instrucción uno o más veces, se traduce de esta manera: a+ => aa*.

Siguiendo con la definición de los objetos NFA y DFA. El porqué se planteó el proyecto con el uso de objetos, se da debido a que en c++ o en el desarrollo de aplicaciones en general, resulta más intuitivo y fácil para el diseño, a la par de ser más amigable para la lectura.

Conforme el cómo se estructuró el NFA. Este se encarga de traducir el postfijo en un contenido para el mapa definido en “structure state”. La principal función de la clase es “constructNFA”, esta llama al resto de funciones, de las cuales se encargan de modificar el mapa para cada una de las instrucciones que la clase recibe (|, * y .), después de definir sus instrucciones individuales, son unidas conforme es necesario. 

Por último tenemos la clase DFA, la cual traduce el mapa y el estado de aceptación del NFA en las transiciones del DFA. Para esto cabe aclarar que la clase NFA, si bien expresa bien las transiciones, suele tener problemas a la hora de contar las mismas, a la par de generar transiciones vacías a la hora de recibir la entrada de caracteres consecutivos (abb => .a.b.b), las cuales generan callejones sin salida, donde deberían de haber transiciones epsilon a la siguiente instancia. Para esto la clase DFA antes de finiquitar la lectura del mapa, se asegura de que no exista más contenido en el mismo, en el caso de que existan, interpreta que la siguiente es la transición a la que tiene que ir la anterior. Esta clase solo tiene una función que construye, que llama “constructDFA”, la cual como aclare, se encarga de traducir las entradas dadas por la clase NFA.

## Conclución:

Como conclusión, puedo decir que este proyecto fue muy retador, principalmente por el algoritmo shunting yard, el cual me costó un tiempo el entender su utilidad y aplicación. Las cuales recaen en una mayor facilidad de entendimiento para la máquina, siendo similar al cambiar de vocabulario empleado al hablar, que al cambiar de idioma.

Con respeto a los autómatas, he logrado entender mejor su funcionamiento, para esto me sirvió mucho el entender las transiciones mediante el diagrama de las mismas, aunque tuve que aprender que no podía representar de la misma forma en el mapa, porque en base del ejemplo dado en las instrucciones, logre entender las equivalencias que tenía que buscar como salida. 

La parte más complicada de hacer fue la depuración y corrección de la clase NFA, debido a los errores explicados, los cuales no pude acabar de resolver, teniendo que adaptar a la clase DFA a trabajar con entradas erróneas.




# Complejidad:

O(3n)


## Explicación:

Cómo explique anteriormente el código tiene tres partes clase, las cuales al encargarse de traducir cuasi linealmente una información dada, primero traduciendo de expresión de regular a postfija,d e post fija al mapa del NFA y por último al DFA, nos queda la unión de tres algoritmos de complejidad O(n), dando en total O(3n).

# Uso:

## Caracteres:

Alfabeto a decisión (se tiene que emplear en la expresión al menos una vez).

| = > or (puede ser un carácter u otro)

Ejemplo: a o b => a|b.

*   => cero o más veces un carácter o expresión 

Ejemplo: a* o (ab)*

+  => uno o más veces un carácter o expresión

Ejemplo: a+ o (ab)+

