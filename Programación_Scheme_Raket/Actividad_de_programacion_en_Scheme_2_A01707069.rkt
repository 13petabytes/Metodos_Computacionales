#lang racket

(display "Ingresa un número para evaluar si es positivo, negativo o 0: ")
(newline)

(define sign
  (lambda (N)
    (cond
      ((= N 0) 0)
      ((< N 0) -1)
      (else 1))))
(sign (read))

(display "Ingresa un pero y altura: ")
(newline)
(define imc
  (lambda (weight height)
         (/ weight (* height height))))
(define peso
  (lambda (weight height)
    (cond
      ((< (imc weight height) 20)"underweight")
      ((and (>= (imc weight height) 20)(< (imc weight height) 25)) "normal")
      ((and (>= (imc weight height) 25)(< (imc weight height) 30)) "obese1")
      ((and (>= (imc weight height) 30)(< (imc weight height) 35)) "obese2")
      (else "obese3"))))
(peso (read) (read))