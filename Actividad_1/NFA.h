#ifndef NFA_H
#define NFA_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <stack>
#include <set>

using namespace std;

struct State {
    int id;
    bool isFinal;
    map<char, vector<State*>> transitions;

    State(int id, bool isFinal = false) : id(id), isFinal(isFinal) {}

    void addTransition(char symbol, State* nextState) {
        transitions[symbol].push_back(nextState);
    }
};

struct Fragment {
    State* start;
    vector<State*> out;

    Fragment(State* start) : start(start) {}
};

class NFA {
private:
    string postfixExp;
    State* startState;
    vector<State*> states;
    int stateCount = 0;

public:
    NFA() : startState(nullptr) {}
    NFA(string postfixExp) : postfixExp(postfixExp), startState(nullptr) {
        constructNFA();
    }
    ~NFA() {
        for (auto state : states) {
            delete state;
        }
    }

    string getPostfixExp() const { return postfixExp; }
    void setPostfixExp(const string& exp) { postfixExp = exp; constructNFA(); }

    Fragment createBasicNFA(char symbol) {
        State* s1 = new State(stateCount++);
        State* s2 = new State(stateCount++, true);
        s1->addTransition(symbol, s2);
        states.push_back(s1);
        states.push_back(s2);
        Fragment fragment(s1);
        fragment.out.push_back(s2);
        return fragment;
    }

    Fragment kleeneNFA(Fragment a) {
        State* start = new State(stateCount++);
        State* end = new State(stateCount++, true);

        cout << "[kleeneNFA] Aplicando Kleene a fragmento con inicio en " << a.start->id << endl;

        start->addTransition('#', a.start);
        start->addTransition('#', end);

        for (State* s : a.out) {
            cout << "  Conectando " << s->id << " al inicio y fin" << endl;
            s->addTransition('#', a.start);
            s->addTransition('#', end);
            s->isFinal = false; 
        }

        states.push_back(start);
        states.push_back(end);

        Fragment fragment(start);
        fragment.out.clear();
        fragment.out.push_back(end);

        cout << "  Estado final único: " << end->id << endl;

        return fragment;
    }

    Fragment concatNFA(Fragment a, Fragment b) {
        cout << "[concatNFA] Concatenando fragmentos:\n";
        cout << "  Últimos estados de A: ";
        for (State* s : a.out) {
            cout << s->id << " ";
        }
        cout << "\n  Primer estado de B: " << b.start->id << endl;

        for (State* s : a.out) {
            s->addTransition('#', b.start);
            s->isFinal = false;
        }

        Fragment fragment(a.start);
        fragment.out = b.out;

        cout << "  Nuevo estado final: " << fragment.out[0]->id << endl;

        return fragment;
    }

    Fragment unionNFA(Fragment a, Fragment b) {
        State* start = new State(stateCount++);
        State* end = new State(stateCount++, true);

        cout << "[unionNFA] Uniendo fragmentos con nuevo estado inicial: " << start->id << endl;

        start->addTransition('#', a.start);
        start->addTransition('#', b.start);

        for (State* s : a.out) {
            s->addTransition('#', end);
            s->isFinal = false;
        }
        for (State* s : b.out) {
            s->addTransition('#', end);
            s->isFinal = false;
        }

        states.push_back(start);
        states.push_back(end);

        Fragment fragment(start);
        fragment.out.clear();
        fragment.out.push_back(end);

        return fragment;
    }

    void constructNFA() {
        stack<Fragment> nfaStack;


        for (char symbol : postfixExp) {
            if (symbol == '|') {
                if (nfaStack.size() < 2) continue;
                Fragment b = nfaStack.top(); nfaStack.pop();
                Fragment a = nfaStack.top(); nfaStack.pop();
                nfaStack.push(unionNFA(a, b));
            } else if (symbol == '.') {
                if (nfaStack.size() < 2) continue;
                Fragment b = nfaStack.top(); nfaStack.pop();
                Fragment a = nfaStack.top(); nfaStack.pop();
                nfaStack.push(concatNFA(a, b));
            } else if (symbol == '*') {
                if (nfaStack.empty()) continue;
                Fragment a = nfaStack.top(); nfaStack.pop();
                nfaStack.push(kleeneNFA(a));
            } else {
                nfaStack.push(createBasicNFA(symbol));
            }
        }

        if (!nfaStack.empty()) {
            startState = nfaStack.top().start;

            cout << "[constructNFA] Definiendo estado final único..." << endl;
            for (State* state : states) {
                state->isFinal = false;
            }
            if (!nfaStack.top().out.empty()) {
                nfaStack.top().out[0]->isFinal = true;
                cout << "  Nuevo estado final: " << nfaStack.top().out[0]->id << endl;
            }
        }
    }
    void printNFA() {
        cout << "🔹 NFA generado:\n";
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
    
            // Solo imprime si hay transiciones definidas
            if (hasTransitions) {
                cout << transitionsOutput << "\n";
            }
        }
        
        cout << "⚡ Estado final: ";
        for (State* state : states) {
            if (state->isFinal) {
                cout << state->id << " ";
            }
        }
        cout << "\n";
    }
    
    map<int, vector<pair<int, char>>> getNFATransitions() {
        map<int, vector<pair<int, char>>> result;
        for (State* state : states) {
            for (auto& [symbol, nextStates] : state->transitions) {
                for (State* nextState : nextStates) {
                    result[state->id].emplace_back(nextState->id, symbol);
                }
            }
        }
        return result;
    }

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

#endif // NFA_H
