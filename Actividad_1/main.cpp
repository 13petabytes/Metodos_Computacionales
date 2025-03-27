#include <iostream>
#include <stack>
#include <string>
#include "NFA.h"
#include "DFA.h"

using namespace std;

// Función que define la precedencia de los operadores
int precedence(char op) {
    if (op == '*') return 3;   // Mayor precedencia (Cerradura de Kleene)
    if (op == '.') return 2;   // Concatenación implícita
    if (op == '|') return 1;   // Unión
    return 0;
}

// Función que convierte una expresión regular a notación posfija
string shuntingYard(const string& regex) {
    stack<char> operators;
    string output;
    bool prevWasChar = false;  // Para detectar concatenaciones implícitas

    for (size_t i = 0; i < regex.size(); i++) {
        char c = regex[i];

        if (isalnum(c)) {  // Si es un carácter (a, b, etc.)
            if (prevWasChar) {
                output += '.'; // Agrega concatenación implícita
            }
            output += c;
            prevWasChar = true;
        } else {
            if (c == '(') {
                if (prevWasChar) {
                    output += '.';  // Agregar concatenación implícita antes del paréntesis
                }
                operators.push(c);
                prevWasChar = false;
            } else if (c == ')') {
                while (!operators.empty() && operators.top() != '(') {
                    output += operators.top();
                    operators.pop();
                }
                if (!operators.empty() && operators.top() == '(') {
                    operators.pop(); // Sacar '('
                }
                prevWasChar = true;
            } else if (c == '|') {
                while (!operators.empty() && precedence(operators.top()) >= precedence(c)) {
                    output += operators.top();
                    operators.pop();
                }
                operators.push(c);
                prevWasChar = false;
            } else if (c == '*') {
                output += c;  // '*' siempre va después de un carácter o grupo
                prevWasChar = true;
            } else if (c == '+') {
                // Convierte `a+` en `aa*`
                if (!output.empty() && isalnum(output.back())) {
                    output += output.back(); // Duplicar el último carácter
                    output += '*';
                } else if (!output.empty() && output.back() == ')') {
                    // Buscar inicio del grupo
                    stack<char> tempStack;
                    string group;
                    
                    // Extraer grupo hasta encontrar '('
                    while (!output.empty() && output.back() != '(') {
                        group = output.back() + group;
                        tempStack.push(output.back());
                        output.pop_back();
                    }

                    if (!output.empty() && output.back() == '(') {
                        output.pop_back(); // Eliminar '('
                        group = '(' + group; // Restaurar el grupo con '('
                    }

                    // Restaurar el grupo en la salida dos veces y agregar '*'
                    output += group + group + "*";
                }
                prevWasChar = true;
            }
        }
    }

    // Vaciar la pila de operadores
    while (!operators.empty()) {
        output += operators.top();
        operators.pop();
    }

    return output;
}

int main() {
    string regex;
    string alfabeto;
    
    cout << "Ingresa el alfabeto: ";
    cin >> alfabeto;

    do {
        cout << "Ingresa una expresión regular: ";
        cin >> regex;

        // Verificar si regex contiene al menos una letra de alfabeto
        bool contieneLetra = false;
        for (char c : alfabeto) {
            if (regex.find(c) != string::npos) {
                contieneLetra = true;
                break;
            }
        }

        // Si contiene una letra del alfabeto, salir del bucle
        if (contieneLetra) break;

        cout << "La expresión debe contener al menos un carácter del alfabeto.\n";

    } while (true);

    cout << "Expresión válida ingresada: " << regex << endl;

    

    string postfix = shuntingYard(regex);
    NFA nfa(postfix);

    cout << "NFA generado:\n";
    nfa.printNFA();

    map<int, vector<pair<int, char>>> nfaTransitions = nfa.getNFATransitions();

    set<int> nfaAcceptingStates = nfa.getNFAAcceptingStates();

    DFA dfa(nfaTransitions, nfaAcceptingStates);
    cout << "\nDFA generado:\n";
    dfa.printDFA();

    return 0;
}
