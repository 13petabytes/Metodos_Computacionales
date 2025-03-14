#ifndef DFA_H
#define DFA_H


#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <string>
#include <algorithm>


using namespace std;


// Estructura que representa un estado del DFA
struct DFAState {
    string name;               // Nombre del estado
    bool isFinal;              // Indica si el estado es de aceptación
    map<char, string> transitions; // Mapa de transiciones: símbolo -> siguiente estado


    // Constructor que inicializa el nombre y el estado de aceptación (por defecto false)
    DFAState(string name, bool isFinal = false) : name(name), isFinal(isFinal) {}
};


// Clase que representa el autómata determinista (DFA)
class DFA {
private:
    map<string, DFAState*> states;          // Mapa de estados del DFA
    string startState;                      // Nombre del estado inicial del DFA
    set<string> acceptingStates;            // Conjunto de estados de aceptación del DFA


public:
    // Constructor que recibe el mapa de transiciones del NFA y los estados de aceptación
    DFA(map<int, vector<pair<int, char>>> nfaMap, set<int> nfaAcceptingStates) {
        constructDFA(nfaMap, nfaAcceptingStates);  // Construcción del DFA a partir del NFA
    }


    // Destructor que limpia la memoria de los estados creados
    ~DFA() {
        for (auto& [name, state] : states) {
            delete state;  // Elimina cada estado del DFA
        }
    }


    // Método para construir el DFA a partir del NFA
    void constructDFA(map<int, vector<pair<int, char>>> nfaMap, set<int> nfaAcceptingStates) {
        queue<set<int>> stateQueue;            // Cola para explorar los conjuntos de estados del NFA
        map<set<int>, string> stateNames;      // Mapa de conjuntos de estados a nombres de estados del DFA
        map<string, set<int>> dfaStates;      // Mapa de nombres de estados del DFA a sus conjuntos de estados NFA
        int stateCounter = 0;                  // Contador para asignar nombres a los estados del DFA
   
        // Función lambda que calcula la clausura epsilon de un conjunto de estados
        auto epsilonClosure = [&](set<int> stateSet) -> set<int> {
            queue<int> q;                    // Cola para explorar los estados en la clausura epsilon
            set<int> closure = stateSet;     // Clausura de estados (inicia con el conjunto dado)
            for (int s : stateSet) q.push(s); // Agrega los estados al inicio de la cola
   
            while (!q.empty()) {
                int current = q.front();
                q.pop();
   
                // Si el estado no tiene transiciones válidas pero el siguiente sí, se salta
                if (nfaMap[current].empty()) {
                    for (auto& [next, transitions] : nfaMap) {
                        if (next > current && !transitions.empty()) {
                            closure.insert(next);
                            q.push(next);
                            break;  // Solo tomar el primer siguiente con transiciones
                        }
                    }
                }
   
                // Explora las transiciones epsilon (símbolo '#')
                for (auto& [next, symbol] : nfaMap[current]) {
                    if (symbol == '#' && closure.find(next) == closure.end()) {
                        closure.insert(next); // Agrega el siguiente estado a la clausura
                        q.push(next);
                    }
                }
            }
            return closure;  // Retorna el conjunto de estados en la clausura epsilon
        };
   
        // Estado inicial del DFA: la clausura epsilon del estado 0 del NFA
        set<int> startClosure = epsilonClosure({0});
        stateQueue.push(startClosure);  // Agrega el estado inicial a la cola
        stateNames[startClosure] = "A"; // Asigna un nombre al primer estado del DFA
        states["A"] = new DFAState("A"); // Crea el estado "A"
        dfaStates["A"] = startClosure;  // Guarda el conjunto de estados en el mapa
   
        // Verifica si el estado inicial del DFA es un estado de aceptación en el NFA
        for (int state : startClosure) {
            if (nfaAcceptingStates.count(state)) {
                states["A"]->isFinal = true;  // Marca el estado "A" como de aceptación
                acceptingStates.insert("A"); // Agrega el estado "A" a los estados de aceptación
                break;
            }
        }
   
        // Explora todos los posibles conjuntos de estados en el DFA
        while (!stateQueue.empty()) {
            set<int> currentSet = stateQueue.front();  // Obtiene el conjunto de estados actual
            stateQueue.pop();                          // Elimina el conjunto de la cola
            string currentName = stateNames[currentSet]; // Nombre del estado actual
   
            map<char, set<int>> moveSets;            // Mapa para almacenar las transiciones por símbolo
   
            // Para cada estado en el conjunto actual, calcula las transiciones por cada símbolo
            for (int state : currentSet) {
                for (auto& [next, symbol] : nfaMap[state]) {
                    if (symbol != '#') {  // Solo considera transiciones no epsilon
                        moveSets[symbol].insert(next);  // Agrega el siguiente estado por el símbolo
                    }
                }
            }
   
            // Para cada conjunto de transiciones calculadas, crea nuevos estados en el DFA
            for (auto& [symbol, nextSet] : moveSets) {
                set<int> closure = epsilonClosure(nextSet);  // Calcula la clausura epsilon del conjunto destino
   
                if (!closure.empty()) {
                    if (stateNames.find(closure) == stateNames.end()) {
                        // Si el conjunto de estados no ha sido visitado, crea un nuevo estado
                        string newStateName = string(1, 'A' + stateCounter++);  // Asigna un nombre único al nuevo estado
                        stateNames[closure] = newStateName; // Mapea el conjunto a su nombre
                        states[newStateName] = new DFAState(newStateName); // Crea el nuevo estado
                        dfaStates[newStateName] = closure; // Guarda el conjunto de estados en el mapa
                        stateQueue.push(closure);  // Agrega el conjunto a la cola para explorar
   
                        // Verifica si el nuevo estado es de aceptación
                        for (int state : closure) {
                            if (nfaAcceptingStates.count(state)) {
                                states[newStateName]->isFinal = true; // Marca el estado como de aceptación
                                acceptingStates.insert(newStateName); // Lo agrega a los estados de aceptación
                                break;
                            }
                        }
                    }
                    // Crea la transición del estado actual al nuevo estado
                    states[currentName]->transitions[symbol] = stateNames[closure];
                }
            }
        }
    }
   
    // Método para imprimir el DFA resultante
    void printDFA() {
        cout << "DFA:\n";
        for (auto& [name, state] : states) {
            cout << name << " => [";
            bool first = true;
            // Imprime las transiciones de cada estado
            for (auto& [symbol, nextState] : state->transitions) {
                if (!first) cout << ", ";
                cout << "('" << nextState << "', '" << symbol << "')";
                first = false;
            }
            cout << "]\n";
        }


        // Imprime los estados de aceptación del DFA
        cout << "Accepting states: [";
        for (auto it = acceptingStates.begin(); it != acceptingStates.end(); ++it) {
            if (it != acceptingStates.begin()) cout << ", ";
            cout << "'" << *it << "'";
        }
        cout << "]\n";
    }
};


#endif // DFA_H




