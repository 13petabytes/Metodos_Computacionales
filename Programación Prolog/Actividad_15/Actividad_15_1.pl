% Caso base: cualquier número elevado a 0 es 1
pow(_, 0, 1) :- !.

% Potencia para exponente positivo
pow(A, B, Res) :-
    integer(B), B > 0,
    B1 is B - 1,
    pow(A, B1, R1),
    Res is A * R1.
