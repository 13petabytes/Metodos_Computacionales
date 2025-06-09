#lang racket

(display "Incerte la cantidad de grados Farenheits a convertir")
(newline)

(define fahrenheit-to-celsius
  (lambda (F)
    (/(* 5(- F 32))9)))
(fahrenheit-to-celsius (read))

(display "Incerte los valores de a, b y c para calcular la ecuación cuadratica")
(newline)

(define roots
  (lambda (a b c)
    (/(- (sqrt(-(* b b)(* 4 a c)))b) (* 2 a))))
(roots (read)(read)(read))
