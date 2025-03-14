
#ifndef NFA_H
#define NFA_H


#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <stack>
#include <set>


using namespace std;


// Estructura que representa un estado en el NFA.
struct State {
    int id;  // Identificador único del estado
    bool isFinal;  // Indica si el estado es de aceptación
    map<char, vector<State*>> transitions;  // Transiciones del estado


    // Constructor de la estructura State, establece el id y si es final
    State(int id, bool isFinal = false) : id(id), isFinal(isFinal) {}


    // Método para agregar una transición desde este estado hacia otro
    void addTransition(char symbol, State* nextState) {
        transitions[symbol].push_back(nextState);
    }
};


// Estructura que representa un fragmento de NFA que se usa en la construcción de NFA.
struct Fragment {
    State* start;  // Estado de inicio del fragmento
    vector<State*> out;  // Estados finales (de salida) del fragmento


    // Constructor que inicializa el estado de inicio del fragmento
    Fragment(State* start) : start(start) {}
};


// Clase que representa un NFA.
class NFA {
private:
    string postfixExp;  // Expresión regular en notación posfija
    State* startState;  // Estado inicial del NFA
    vector<State*> states;  // Lista de todos los estados del NFA
    int stateCount = 0;  // Contador de estados para asignar ids únicos


public:
    // Constructor vacío del NFA
    NFA() : startState(nullptr) {}


    // Constructor que recibe una expresión regular en notación posfija
    NFA(string postfixExp) : postfixExp(postfixExp), startState(nullptr) {
        constructNFA();  // Llama a la función para construir el NFA
    }


    // Destructor que elimina los estados del NFA
    ~NFA() {
        for (auto state : states) {
            delete state;  // Libera memoria de cada estado
        }
    }


    // Obtiene la expresión regular en notación posfija
    string getPostfixExp() const { return postfixExp; }


    // Establece la expresión regular en notación posfija y reconstruye el NFA
    void setPostfixExp(const string& exp) { postfixExp = exp; constructNFA(); }


    // Crea un NFA básico para un símbolo de la expresión regular
    Fragment createBasicNFA(char symbol) {
        State* s1 = new State(stateCount++);  // Estado inicial
        State* s2 = new State(stateCount++, true);  // Estado final
        s1->addTransition(symbol, s2);  // Agrega la transición del símbolo
        states.push_back(s1);  // Agrega los estados al NFA
        states.push_back(s2);
        Fragment fragment(s1);  // Crea un fragmento con el estado inicial
        fragment.out.push_back(s2);  // Establece el estado final
        return fragment;
    }


    // Crea un NFA aplicando la operación de Kleene (*) a un fragmento dado
    Fragment kleeneNFA(Fragment a) {
        State* start = new State(stateCount++);  // Estado de inicio
        State* end = new State(stateCount++, true);  // Estado final


        // Agrega transiciones para la operación de Kleene
        start->addTransition('#', a.start);
        start->addTransition('#', end);


        // Conecta los estados de salida del fragmento a inicio y final
        for (State* s : a.out) {
            s->addTransition('#', a.start);
            s->addTransition('#', end);
            s->isFinal = false;  // Los estados de salida ya no son finales
        }


        states.push_back(start);  // Agrega los nuevos estados al NFA
        states.push_back(end);


        Fragment fragment(start);  // Crea un fragmento con el estado inicial
        fragment.out.clear();  // Limpia los estados de salida
        fragment.out.push_back(end);  // Establece el nuevo estado final


        return fragment;  // Devuelve el fragmento modificado
    }


    // Crea un NFA concatenando dos fragmentos
    Fragment concatNFA(Fragment a, Fragment b) {
        // Conecta las salidas de 'a' con el inicio de 'b'
        for (State* s : a.out) {
            s->addTransition('#', b.start);
            s->isFinal = false;  // Los estados de salida de 'a' no son finales
        }


        Fragment fragment(a.start);  // Crea un fragmento con el estado inicial de 'a'
        fragment.out = b.out;  // Establece los estados de salida de 'b'


        return fragment;  // Devuelve el fragmento concatenado
    }


    // Crea un NFA que representa la unión de dos fragmentos
    Fragment unionNFA(Fragment a, Fragment b) {
        State* start = new State(stateCount++);  // Nuevo estado de inicio
        State* end = new State(stateCount++, true);  // Nuevo estado final


        // Conecta el nuevo estado de inicio con los inicios de ambos fragmentos
        start->addTransition('#', a.start);
        start->addTransition('#', b.start);


        // Conecta los estados de salida de ambos fragmentos al nuevo estado final
        for (State* s : a.out) {
            s->addTransition('#', end);
            s->isFinal = false;
        }
        for (State* s : b.out) {
            s->addTransition('#', end);
            s->isFinal = false;
        }


        states.push_back(start);  // Agrega los nuevos estados al NFA
        states.push_back(end);


        Fragment fragment(start);  // Crea un fragmento con el nuevo estado inicial
        fragment.out.clear();  // Limpia los estados de salida
        fragment.out.push_back(end);  // Establece el nuevo estado final


        return fragment;  // Devuelve el fragmento unido
    }


    // Función que construye el NFA a partir de la expresión regular en notación posfija
    void constructNFA() {
        stack<Fragment> nfaStack;  // Pila para almacenar los fragmentos del NFA


        // Procesa cada símbolo en la expresión regular
        for (char symbol : postfixExp) {
            if (symbol == '|') {
                // Si es el operador '|', realiza la unión de dos fragmentos
                if (nfaStack.size() < 2) continue;
                Fragment b = nfaStack.top(); nfaStack.pop();
                Fragment a = nfaStack.top(); nfaStack.pop();
                nfaStack.push(unionNFA(a, b));
            } else if (symbol == '.') {
                // Si es el operador '.', realiza la concatenación de dos fragmentos
                if (nfaStack.size() < 2) continue;
                Fragment b = nfaStack.top(); nfaStack.pop();
                Fragment a = nfaStack.top(); nfaStack.pop();
                nfaStack.push(concatNFA(a, b));
            } else if (symbol == '*') {
                // Si es el operador '*', aplica la operación de Kleene a un fragmento
                if (nfaStack.empty()) continue;
                Fragment a = nfaStack.top(); nfaStack.pop();
                nfaStack.push(kleeneNFA(a));
            } else {
                // Si es un símbolo literal, crea un NFA básico para él
                nfaStack.push(createBasicNFA(symbol));
            }
        }


        // Al finalizar, establece el estado inicial y el final del NFA
        if (!nfaStack.empty()) {
            startState = nfaStack.top().start;
            for (State* state : states) {
                state->isFinal = false;
            }
            if (!nfaStack.top().out.empty()) {
                nfaStack.top().out[0]->isFinal = true;
            }
        }
    }


    // Imprime el NFA generado en formato legible
    void printNFA() {
        for (State* state : states) {
            bool hasTransitions = false;
            string transitionsOutput = to_string(state->id) + " => [";
            for (auto& [symbol, nextStates] : state->transitions) {
                for (State* nextState : nextStates) {
                    if (hasTransitions) transitionsOutput += ", ";
                    transitionsOutput += "(" + to_string(nextState->id) + ", '" + symbol + "')";
                    hasTransitions = true;
                }
            }
            transitionsOutput += "]";
            if (hasTransitions) {
                cout << transitionsOutput << "\n";
            }
        }


        // Imprime los estados de aceptación del NFA
        cout << "⚡ Estado final: ";
        for (State* state : states) {
            if (state->isFinal) {
                cout << state->id << " ";
            }
        }
        cout << "\n";
    }


    // Devuelve las transiciones del NFA como un mapa de estados y transiciones
    map<int, vector<pair<int, char>>> getNFATransitions() {
        map<int, vector<pair<int, char>>> result;
        for (State* state : states) {
            for (auto& [symbol, nextStates] : state->transitions) {
                for (State* nextState : nextStates) {
                    result[state->id].push_back({nextState->id, symbol});
                }
            }
        }
        return result;
    }


    // Devuelve los estados de aceptación del NFA
    set<int> getNFAAcceptingStates() {
        set<int> acceptingStates;
        for (State* state : states) {
            if (state->isFinal) {
                acceptingStates.insert(state->id);
            }
        }
        return acceptingStates;
    }
};


#endif  // NFA_H
