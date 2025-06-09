#lang racket
(display "Incerte una lista para leerla y dar los positivos:")
(newline)

(define how-many-positives
  (lambda (lst)
    (cond
      ((null? lst) 0)
      ((positive? (car lst)) (+ 1 (how-many-positives (cdr lst))))
      (else (how-many-positives (cdr lst)))))) 

(how-many-positives (read))


(display "Incerte un numero y una lista para leerla y decirle cuantas veces se repite:")
(newline)

(define count
  (lambda (b lst)
    (cond
      ((null? lst) 0)
      ((equal? (car lst) b) (+ 1 (count b (cdr lst)))) 
      (else (count b (cdr lst))))))

(count (read)(read))