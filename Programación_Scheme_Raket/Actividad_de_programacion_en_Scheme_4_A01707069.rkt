#lang racket
(display "Ingresa un numero y su potencia:")
(newline)

(define pow
  (lambda (a b)
    (if (= b 0)
        1
        (* a (pow a (- b 1))))))

(pow (read)(read))


(display "Ingresa un numero, y otra para dividir al primer numero:")
(newline)
(define dividing-by-subtraction
  (lambda (dividend divisor)
    (if (< dividend divisor)
        0
        (+ 1 (dividing-by-subtraction (- dividend divisor) divisor)))))

(dividing-by-subtraction (read)(read))