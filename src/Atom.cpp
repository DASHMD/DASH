
#include "Atom.h"
#include "boost_for_export.h"
namespace py = boost::python;


void Atom::setPos(Vector &x) {
    isChanged = true;
    pos = x;
}
Vector Atom::getPos() {
    return pos;
}


void Atom::setVel(Vector &x) {
    isChanged = true;
    vel = x;
}
Vector Atom::getVel() {
    return vel;
}


void Atom::setForce(Vector &x) {
    isChanged = true;
    force = x;
}
Vector Atom::getForce() {
    return force;
}


std::string Atom::getType() {
    return handles->at(type);
}

void Atom::setNDF(int n) {
    ndf = n;
}

int Atom::getNDF() {
    return ndf;
}

void Atom::setBeadPos(int n, int nPerRingPoly, std::vector<Vector> &xsNM) {
    float sqrt2           = sqrt(2.0);
    float invP            = 1.0 / (float) nPerRingPoly;
    float twoPiInvP       = 2.0f * M_PI * invP;
    float invSqrtP        = sqrtf(invP);
    int halfP = nPerRingPoly/2;

    // k = 0
    Vector xn = xsNM[0];
    
    // k = halfP
    // xn += xsNM[halfP]*(-1)**n, for n = 1,...,P
    if ( n % 2 == 0 ) {
      xn += xsNM[halfP];}
    else {
      xn -= xsNM[halfP];}
    
    // k = 1,...,P/2-1; n = 1,...,P
    for (int k = 1; k < halfP; k++) {
      float  cosval = cos(twoPiInvP * k * n); // cos(2*pi*k*n/P)
      xn += xsNM[k] * sqrt2 * cosval;
    }
    
    // k = P/2+1,...,P-1; n = 1,...,P
    for (int k = halfP+1; k < nPerRingPoly; k++) {
      float  sinval = sin(twoPiInvP * k * n); // cos(2*pi*k*n/P)
      xn += xsNM[k] * sqrt2 * sinval;
    }
    
    // replace evolved back-transformation
    pos = xn*invSqrtP;
}

//void Atom::setBeadVel(int nPerRingPoly, float betaP) {
//    Vector vn;
//}

void export_Atom () { 
    py::class_<Atom>("Atom", py::no_init)
        .def_readonly("id", &Atom::id)
        //.add_property("pos", &Atom::getPos, &Atom::setPos)
        .def_readwrite("pos", &Atom::pos)
        .def_readwrite("vel", &Atom::vel)
        .def_readwrite("force", &Atom::force)
        //.add_property("vel", &Atom::getVel, &Atom::setVel)
        //.add_property("force", &Atom::getForce, &Atom::setForce)
        .def_readwrite("groupTag", &Atom::groupTag)
        .def_readwrite("mass", &Atom::mass)
        .def_readwrite("q", &Atom::q)
        .def_readwrite("ndf",&Atom::ndf)
        .add_property("type", &Atom::getType)
        .def("kinetic", &Atom::kinetic)
        .def_readwrite("isChanged", &Atom::isChanged)
        ;

}


