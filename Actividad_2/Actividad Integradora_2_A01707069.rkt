#lang racket

;; Manejo de Automata y entradas

(define (validate automata cadenas)
  (define estados (first automata))      ; 1. Definir los estados del autómata, complejidad O(1)
  (define alfabeto (second automata))    ; 2. Definir el alfabeto, complejidad O(1)
  (define transiciones (third automata)) ; 3. Definir las transiciones, complejidad O(1)
  (define inicio (fourth automata))      ; 4. Definir el estado inicial, complejidad O(1)
  (define finales (fifth automata))      ; 5. Definir los estados finales, complejidad O(1)

  ;; Construcción de la tabla de transiciones, complejidad O(T), donde T es el numero de numero de transisiones
  (define tabla-transiciones
    (map (lambda (t)
           (cons (list (first t) (second t)) (third t)))
         transiciones))

  ;; Función para obtener el siguiente estado, complejidad por llamada O(T), donde T es
  (define siguiente-estado
    (lambda (estado simbolo)
      (let ((par (assoc (list estado simbolo) tabla-transiciones)))  ; Buscar transición
        (if par (cdr par) #f))))                                     ; Si existe la transición, devuelve el siguiente estado, sino devuelve #f

  ;; Función para evaluar si una cadena es aceptada, complejidad de O(L * T), donde L corresponde a la longitud de la cadena, y T a la complejidad de la función siguiente-etado
  (define acepta?
    (lambda (cadena)
      (letrec ((recorrer
                (lambda (estado restante)
                  (if (null? restante)                                            ; Si no hay más símbolos
                      (if (member estado finales) #t #f)                          ; Verifica si el estado actual es final
                      (let ((siguiente (siguiente-estado estado (car restante)))) ; Obtiene el siguiente estado
                        (if siguiente
                            (recorrer siguiente (cdr restante))                   ; Recursión con el siguiente estado
                            #f))))))                                              ; Si no hay transición, termina con #f
        (recorrer inicio cadena))))                                               ; Comienza desde el estado inicial

  ;; Llamadas a la función acepta? en función de numero de cadenas a evaluar, complejidad O(C * L * T), donde  L * T representa a la función acepta? y C el numero de cadenas
  (map (lambda (c) (acepta? c)) cadenas))

;; Manejo de usuario
(display "Inserte el cuerpo del automata a trabajar, como una lista profunda:\n") 
(newline)

;; Cuerpo del automata
(define automaton (read)) ; Complejidad de los "read" O(N), donde N es el tamño del imput

(display "Inserte las cadenas de entrada, como lista profunda:\n")
(newline)

;; Entradas
(define inputs (read))

(display "Resultados:\n")
(newline)

;; Validaciones
(display (validate automaton inputs)) ; Complejidad del "display" O(B), donde B es el numero de Booleanos o respunta a devolver
