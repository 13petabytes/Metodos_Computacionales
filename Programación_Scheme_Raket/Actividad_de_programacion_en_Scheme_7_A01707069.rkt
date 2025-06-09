#lang racket
(display "Incerte una lista para modificarla:")
(newline)

(define enlist
  (lambda (lst)
    (map (lambda (x) (list x)) lst)))

(enlist (read))

(display "Incerte una lista con listas de dos elementos, para invertir los elementos de las listas pequeñas:")
(newline)

(define invert-pairs
  (lambda (lst)
    (map (lambda (pair) (list (cadr pair) (car pair))) lst)))

(invert-pairs (read))

(display "Incerte una lista para invertir su contenido:")
(newline)

(define deep-reverse
  (lambda (lst)
    (if (null? lst)
        '()
        (append (deep-reverse (cdr lst))
                (list (if (list? (car lst))
                          (deep-reverse (car lst))
                          (car lst)))))))

(deep-reverse (read))

(display "Incerte una lista para agrupar a los elementos repetidos:")
(newline)

(define pack
  (lambda (lst)
    (if (null? lst)
        '()
        (let loop ((remaining lst) (current (list (car lst))) (result '()))
          (if (null? (cdr remaining))
              (reverse (cons current result))
              (if (equal? (car remaining) (cadr remaining))
                  (loop (cdr remaining) (cons (cadr remaining) current) result)
                  (loop (cdr remaining) (list (cadr remaining)) (cons current result))))))))


(pack (read))

(display "Incerte una lista para indicar la cantidad que se repite cada elemento:")
(newline)

(define encode
  (lambda (lst)
    (map (lambda (sublist) (list (length sublist) (car sublist))) (pack lst))))

(encode (read))