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

struct DFAState {
    string name;
    bool isFinal;
    map<char, string> transitions;

    DFAState(string name, bool isFinal = false) : name(name), isFinal(isFinal) {}
};

class DFA {
private:
    map<string, DFAState*> states;
    string startState;
    set<string> acceptingStates;

public:
    DFA(map<int, vector<pair<int, char>>> nfaMap, set<int> nfaAcceptingStates) {
        constructDFA(nfaMap, nfaAcceptingStates);
    }

    ~DFA() {
        for (auto& [name, state] : states) {
            delete state;
        }
    }

    void constructDFA(map<int, vector<pair<int, char>>> nfaMap, set<int> nfaAcceptingStates) {
        queue<set<int>> stateQueue;
        map<set<int>, string> stateNames;
        map<string, set<int>> dfaStates;
        int stateCounter = 0;
    
        // Función para calcular la clausura epsilon y detectar estados vacíos
        auto epsilonClosure = [&](set<int> stateSet) -> set<int> {
            queue<int> q;
            set<int> closure = stateSet;
            for (int s : stateSet) q.push(s);
    
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
    
                for (auto& [next, symbol] : nfaMap[current]) {
                    if (symbol == '#' && closure.find(next) == closure.end()) {
                        closure.insert(next);
                        q.push(next);
                    }
                }
            }
            return closure;
        };
    
        // Estado inicial del DFA (clausura epsilon del estado 0 del NFA)
        set<int> startClosure = epsilonClosure({0});
        stateQueue.push(startClosure);
        stateNames[startClosure] = "A";
        states["A"] = new DFAState("A");
        dfaStates["A"] = startClosure;
    
        // Verificar si el estado inicial del DFA es de aceptación
        for (int state : startClosure) {
            if (nfaAcceptingStates.count(state)) {
                states["A"]->isFinal = true;
                acceptingStates.insert("A");
                break; 
            }
        }
    
        while (!stateQueue.empty()) {
            set<int> currentSet = stateQueue.front();
            stateQueue.pop();
            string currentName = stateNames[currentSet];
    
            map<char, set<int>> moveSets;
    
            for (int state : currentSet) {
                for (auto& [next, symbol] : nfaMap[state]) {
                    if (symbol != '#') {
                        moveSets[symbol].insert(next);
                    }
                }
            }
    
            for (auto& [symbol, nextSet] : moveSets) {
                set<int> closure = epsilonClosure(nextSet);
    
                if (!closure.empty()) {
                    if (stateNames.find(closure) == stateNames.end()) {
                        string newStateName = string(1, 'A' + stateCounter++);
                        stateNames[closure] = newStateName;
                        states[newStateName] = new DFAState(newStateName);
                        dfaStates[newStateName] = closure;
                        stateQueue.push(closure);
    
                        for (int state : closure) {
                            if (nfaAcceptingStates.count(state)) {
                                states[newStateName]->isFinal = true;
                                acceptingStates.insert(newStateName);
                                break;
                            }
                        }
                    }
                    states[currentName]->transitions[symbol] = stateNames[closure];
                }
            }
        }
    }
             
        void printDFA() {
        cout << "DFA:\n";
        for (auto& [name, state] : states) {
            cout << name << " => [";
            bool first = true;
            for (auto& [symbol, nextState] : state->transitions) {
                if (!first) cout << ", ";
                cout << "('" << nextState << "', '" << symbol << "')";
                first = false;
            }
            cout << "]\n";
        }

        cout << "Accepting states: [";
        for (auto it = acceptingStates.begin(); it != acceptingStates.end(); ++it) {
            if (it != acceptingStates.begin()) cout << ", ";
            cout << "'" << *it << "'";
        }
        cout << "]\n";
    }
};

#endif // DFA_H
