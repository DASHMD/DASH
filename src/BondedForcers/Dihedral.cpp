#include "Dihedral.h"
#include "boost_for_export.h"
#include "array_indexing_suite.hpp"
namespace py = boost::python;
DihedralOPLS::DihedralOPLS(Atom *a, Atom *b, Atom *c, Atom *d, double coefs_[4], int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    ids[3] = d->id;
    for (int i=0; i<4; i++) {
        coefs[i] = coefs_[i];
    }
    type = type_;
}

DihedralOPLS::DihedralOPLS(double coefs_[4], int type_) {
    for (int i=0; i<4; i++) {
        ids[i] = -1;
    }
    for (int i=0; i<4; i++) {
        coefs[i] = coefs_[i];
    }
    type = type_;
}



DihedralCHARMM::DihedralCHARMM(Atom *atomA, Atom *atomB, Atom *atomC, Atom *atomD, double k_, int n_, double d_,  int type_) {
    ids[0] = atomA->id;
    ids[1] = atomB->id;
    ids[2] = atomC->id;
    ids[3] = atomD->id;
    k = k_;
    n = n_;
    d = d_;
    type = type_;
}

DihedralCHARMM::DihedralCHARMM(double k_, int n_, double d_, int type_) {
    for (int i=0; i<4; i++) {
        ids[i] = -1;
    }
    k = k_;
    n = n_;
    d = d_;
    type = type_;
}


void Dihedral::takeIds(Dihedral *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}


void DihedralGPU::takeIds(Dihedral *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}

DihedralOPLSType::DihedralOPLSType(DihedralOPLS *dihedral) {
    for (int i=0; i<4; i++) {
        coefs[i] = dihedral->coefs[i];
    }
}
DihedralCHARMMType::DihedralCHARMMType(DihedralCHARMM *dihedral) {
    k = dihedral->k;
    n = dihedral->n;
    d = dihedral->d;
}


std::string DihedralOPLS::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c='" << ids[2] << "' atomID_d='" << ids[3] << "' coef_a='" << coefs[0]<< "' coef_b='" << coefs[1] << "' coef_c='" << coefs[2] << "' coef_d='" << coefs[3] << "'/>\n";
  return ss.str();
}

std::string DihedralOPLSType::getInfoString() {
    std::stringstream ss;
    ss << " coef_a='" << coefs[0]<< "' coef_b='" << coefs[1] << "' coef_c='" << coefs[2] << "' coef_d='" << coefs[3];
    return ss.str();
}



std::string DihedralCHARMM::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c='" << ids[2] << "' atomID_d='" << ids[3] << " k='" << k << "' n='" << n << "' d='" << d << "'/>\n";
  return ss.str();
}

std::string DihedralCHARMMType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' n='" << n << "' d='" << d;
  return ss.str();
}
bool DihedralOPLSType::operator==(const DihedralOPLSType &other) const {
    for (int i=0; i<4; i++) {
        if (coefs[i] != other.coefs[i]) {
            return false;
        }
    }
    return true;
}

bool DihedralCHARMMType::operator==(const DihedralCHARMMType &other) const {
    return other.k == k and other.d == d and other.n == n;
}



void export_Dihedrals() {
    py::class_<DihedralOPLS, SHARED(DihedralOPLS)> ( "SimDihedralOPLS", py::init<>())
        .def_readwrite("type", &DihedralOPLS::type)
        .def_readonly("coefs", &DihedralOPLS::coefs)
        .def_readonly("ids", &DihedralOPLS::ids)

    ;
    py::class_<DihedralCHARMM, SHARED(DihedralCHARMM)> ( "SimDihedralCHARMM", py::init<>())
        .def_readwrite("type", &DihedralCHARMM::type)
        .def_readwrite("k", &DihedralCHARMM::k)
        .def_readwrite("n", &DihedralCHARMM::n)
        .def_readwrite("d", &DihedralCHARMM::d)
        .def_readonly("ids", &DihedralCHARMM::ids)

    ;

}
