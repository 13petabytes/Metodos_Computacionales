#lang racket
(display "Incerte una lista de numeros:")
(newline)

(define sumatoria
  (lambda (datos)
    (let loop ((datos datos) (acum 0))
      (if (null? datos)
          acum
          (loop (cdr datos) (+ acum (car datos))))))
)

(sumatoria (read))

(display "Incerte una lista de numeros:")
(newline)

(define incrementa
  (lambda (datos)
    (let loop ((datos datos) (acum '()))
      (if (null? datos)
          (reverse acum)
          (loop (cdr datos) (cons (+ 1 (car datos)) acum)))))
)

(incrementa (read))