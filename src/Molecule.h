#pragma once
#ifndef MOLECULE_H

#include <vector>
#include "Vector.h"

class State;

//MAKE IT SO BY DEFAULT MOLEC DOES TOPO SEARCH TO INCLUDE EVERYTHING IT'S CONNECT TO`
void export_Molecule();

class Molecule {
private:
    State *state;
public:
    std::vector<int> ids;
    Molecule(State *, std::vector<int> &ids_);
    void translate(Vector &);
    void rotate(Vector axis, double theta);
    void rotateRandom();
    Vector COM();
    bool operator==(const Molecule &other) {
        return ids == other.ids;
    }
	void unwrap();
    double dist(Molecule &);
    Vector size();
};




#endif
