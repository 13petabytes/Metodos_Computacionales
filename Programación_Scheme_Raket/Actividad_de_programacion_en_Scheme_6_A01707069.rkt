#lang racket
(display "Incerte una lista paraduplicarla")
(newline)
(define duplicate 
  (lambda (lst)
    (if (null? lst)
        '()
        (cons (car lst) (cons (car lst) (duplicate (cdr lst)))))))

(duplicate (read))


(display "Incerte una lista darle los numeros positivos")
(newline)
(define positives 
  (lambda (lst)
    (filter (lambda (x) (> x 0)) lst)))

(positives (read))

(display "Incerte una lista para detectar si hay numeros")
(newline)

(define list-of-symbols? 
  (lambda (lst)
    (cond ((null? lst) #t)
          ((symbol? (car lst)) (list-of-symbols? (cdr lst)))
          (else #f))))

(list-of-symbols? (read))

(display "Incerte dos datos y una lista para intercambiarlos")
(newline)

(define swapper 
  (lambda (a b lst)
    (map (lambda (x)
           (cond ((equal? x a) b)
                 ((equal? x b) a)
                 (else x))) lst)))

(swapper (read)(read)(read))

(display "Incerte dos listas para regresarle el producto punto")
(newline)

(define dot-product 
  (lambda (a b)
    (if (or (null? a) (null? b))
        0
        (+ (* (car a) (car b)) (dot-product (cdr a) (cdr b))))))

(dot-product(read)(read))
