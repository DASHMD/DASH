#pragma once
#ifndef ATOM_H
#define ATOM_H

#include <vector>
#include <string.h>
#include "Vector.h"
#include "OffsetObj.h"

class Atom;


void export_Atom();

class Atom {
public:
    Vector pos;
    Vector vel;
    Vector force;
    //Vector posAtNeighborListing; // can do this elsewhere

    double mass;
    double q;
    int type;  
    int id;
    uint32_t groupTag;
    bool isChanged;
    std::vector<std::string> *handles;

    Atom (std::vector<std::string> *handles_) 
        : mass(0), type(-1), id(-1), handles(handles_)
    {};
    Atom(Vector pos_, int type_, int id_, double mass_, double q_, std::vector<std::string> *handles_)
        : pos(pos_), mass(mass_), q(q_), type(type_), id(id_), groupTag(1), handles(handles_)
    {   }

    bool operator==(const Atom &other) {
        return id == other.id;
    }
    bool operator!=(const Atom &other) {
        return id != other.id;
    }

    double kinetic() {
        return 0.5 * mass * vel.lenSqr();
    }

    void setPos(Vector &x);
    Vector getPos();

    void setVel(Vector &x);
    Vector getVel();

    void setForce(Vector &x);
    Vector getForce();

    std::string getType();

    void setBeadPos(int n, int nPerRingPoly, std::vector<Vector> &xsNM);
    // displaces the position of a bead based on free ring-polymer distribution

    //void setBeadVel(int nPerRingPoly, float betaP);


};

#endif
