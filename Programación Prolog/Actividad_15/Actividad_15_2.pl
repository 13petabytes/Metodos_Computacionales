% Casos base
fib(0, 0) :- !.
fib(1, 1) :- !.

% Caso recursivo
fib(N, X) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, X1),
    fib(N2, X2),
    X is X1 + X2.
