#pragma once
#ifndef BOUNDS_H
#define BOUNDS_H

#include <stdio.h>
#include <assert.h>

#include "Python.h"

#include "globalDefs.h"
#include "Atom.h"
#include "BoundsGPU.h"
#include "Vector.h"
#include "boost_for_export.h"

void export_Bounds();

class State;

class Bounds {
public:
    State *state;
    bool isSet;
    Vector lo;
    Vector rectComponents; //xx, yy, zz components of the bounds.  lo + rectComponents = upper corner of the box when box is rectangular
    Bounds();
    Bounds(State *); //constructor to be called within state's constructor.  Initializes values to some default values which can be used to check if values are set
    
    Bounds(SHARED(State) state_, Vector lo_, Vector hi_) {
        init(state_.get(), lo_, hi_);
    }
    Bounds(State *state_, Vector lo_, Vector hi_) {
        init(state_, lo_, hi_);
    }
    Bounds(BoundsGPU bounds) {
        init(nullptr, Vector(bounds.lo), Vector(bounds.lo + bounds.rectComponents));
    }
    void setHiPy(Vector &v);
    Vector getHiPy();

    void setLoPy(Vector &v);
    Vector getLoPy();

    Vector getRectComponentsPy();
    void init(State *, Vector lo_, Vector hi_);

    bool operator==(const Bounds &other) {
        return lo == other.lo and rectComponents == rectComponents;
    }
    bool operator!=(const Bounds &other) {
        return !(*this == other);
    }

    Bounds copy() {
        return *this;
    }
    Vector wrap(Vector v);
    bool isInitialized();
    void set(BoundsGPU &b) {
        lo = Vector(b.lo);
        rectComponents = Vector(b.rectComponents);
        /*
        lo = Vector(b.lo);
        //hi = lo;
        trace = Vector();
        for (int i=0; i<3; i++) {
            sides[i] = Vector(b.sides[i]);
            trace += sides[i];
        }
        hi += trace;
        */
    }
    void set(Bounds &b) {
        lo = b.lo;
        rectComponents = b.rectComponents;
        /*
        lo = b.lo;
        hi = b.hi;
        for (int i=0; i<3; i++) {
            sides[i] = b.sides[i];
        }
        trace = b.trace;
        */
    }
    void setSides() {
        /*
        trace = hi-lo;
        for (int i=0; i<3; i++) {
            Vector v = Vector(0, 0, 0);
            v[i] = trace[i];
            sides[i] = v;
        }
        */
    }
    void setPython(Bounds &b) {
        /*
        set(b);
        */
    }

    void handle2d();
    BoundsGPU makeGPU();
    bool vectorInBounds(Vector &);
    bool atomInBounds(Atom &);
    double volume();

  //  Bounds unskewed();
    Vector minImage(Vector v);

};

//SHARED(Bounds) BoundsCreateSkew(  figure out how to create laters

#endif
