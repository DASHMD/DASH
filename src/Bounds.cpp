
#include <string>
#include <iostream>

#include "Bounds.h"
#include "State.h"
#include "AtomParams.h"

using namespace std;
#define DEFAULT_VAL -1
Bounds::Bounds() {
    state = nullptr;
}
Bounds::Bounds(State *state_) {
    state = state_;
    rectComponents = Vector(DEFAULT_VAL, DEFAULT_VAL, DEFAULT_VAL);
}

void Bounds::init(State *state_, Vector lo_, Vector hi_) {
    state = state_;
    lo = lo_;
    rectComponents = hi_ - lo;

}

bool Bounds::isInitialized() {
    return rectComponents != Vector(DEFAULT_VAL, DEFAULT_VAL, DEFAULT_VAL);
}
void Bounds::handle2d() {
    if (state->is2d) {
        lo[2] = -0.5;
        rectComponents[2] = 1.0;
    }
}

BoundsGPU Bounds::makeGPU() {
    bool *periodic = state->periodic;
    return BoundsGPU(lo.asFloat3(), rectComponents.asFloat3(), make_float3((int) periodic[0], (int) periodic[1], (int) periodic[2]));
}

bool Bounds::vectorInBounds(Vector &v) {
    Vector hi = lo + rectComponents;
    for (int i=0; i<3; i++) {
        if (not (v[i] >= lo[i] and v[i] <= hi[i])) {
            return false;
        }
    }
    return true;
}
bool Bounds::atomInBounds(Atom &a) {
    return vectorInBounds(a.pos);
}

double Bounds::volume() {
    return rectComponents.prod();
}


/*
bool Bounds::skew(Vector skewBy) {
    skewBy[2] = 0;
    sides[0][1] += skewBy[1];
    sides[1][0] += skewBy[0];
    trace += skewBy;
    return true;
}
double Bounds::getSkewX() {
    return atan2(sides[0][1], sides[0][0]);
}

double Bounds::getSkewY() {
    return atan2(sides[1][0], sides[1][1]);
}
*/

/*
Bounds Bounds::unskewed() {
    Vector hiNew = lo;
    for (int i=0; i<3; i++) {
        hiNew[i] = lo[i] + sides[i][i];
    }
    return Bounds(state, lo, hiNew);
}

bool Bounds::isSkewed() {
    for (int i=0; i<3; i++) {
        Vector test(1, 1, 1);
        test[i] = 0;
        Vector res = sides[i] * test;
        if (res.abs() > VectorEps) {
            return true;
        }
    }
    return false;
}
*/
Vector Bounds::minImage(Vector v) {
    for (int i=0; i<3; i++) {
        int img = round(v[i] / rectComponents[i]);
        v[i] -= rectComponents[i] * (state->periodic[i] * img);
    }
    return v;
}
Vector Bounds::wrap(Vector v) {
    Vector dVec = v - lo;
    VectorInt nImage = dVec / rectComponents; 
    v -= nImage * rectComponents;
    return v + lo;
}

void Bounds::setHiPy(Vector &v) {
    rectComponents = v - lo;
}
Vector Bounds::getHiPy() {
    return lo + rectComponents;
}

//keeping hi constant
void Bounds::setLoPy(Vector &v) {
    Vector dlo = v - lo;
    rectComponents -= dlo;
    lo = v;
}
Vector Bounds::getLoPy() {
    return lo;
}
Vector Bounds::getRectComponentsPy() {
    return rectComponents;
}

void export_Bounds() {
    boost::python::class_<Bounds, SHARED(Bounds)>(
        "Bounds",
        boost::python::init<SHARED(State), Vector, Vector>(
            boost::python::args("state", "lo", "hi")
        )
    )
    .def("vectorInBounds", &Bounds::vectorInBounds)
    .def("atomInBounds", &Bounds::atomInBounds)
    .def("copy", &Bounds::copy)
    .def("set", &Bounds::setPython)
    .def("minImage", &Bounds::minImage)
    .def("volume", &Bounds::volume)
    .add_property("lo", &Bounds::getLoPy, &Bounds::setLoPy)
    .add_property("hi", &Bounds::getHiPy, &Bounds::setHiPy)
    .add_property("rectComponents", &Bounds::getRectComponentsPy)
    ;
}
