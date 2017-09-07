#include "Improper.h"
#include "boost_for_export.h"
namespace py = boost::python;



void Improper::takeIds(Improper *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}



void ImproperGPU::takeIds(Improper *other) {
    for (int i=0; i<4; i++) {
        ids[i] = other->ids[i];
    }
}






ImproperHarmonic::ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k_, double thetaEq_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    ids[3] = d->id;
    k = k_;
    thetaEq = thetaEq_;
    type = type_;

}
ImproperHarmonic::ImproperHarmonic(double k_, double thetaEq_, int type_) {
    for (int i=0; i<4; i++) {
        ids[i] = -1;
    }
    k = k_;
    thetaEq = thetaEq_;
    type = type_;

}

ImproperHarmonicType::ImproperHarmonicType(ImproperHarmonic *imp) {
    k = imp->k;
    thetaEq = imp->thetaEq;
}
bool ImproperHarmonicType::operator==(const ImproperHarmonicType &other) const {
    return k == other.k and thetaEq == other.thetaEq;
}

std::string ImproperHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' thetaEq='" << thetaEq << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID\
_c='" << ids[2] << "' atomID_d='" << ids[3] << "'/>\n";
  return ss.str();
}

std::string ImproperHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' thetaEq='" << thetaEq;
  return ss.str();
}




ImproperCVFF::ImproperCVFF(Atom *a_, Atom *b_, Atom *c_, Atom *d_, double k_, int dParam_, int n_, int type_) {
    ids[0] = a_->id;
    ids[1] = b_->id;
    ids[2] = c_->id;
    ids[3] = d_->id;
    k = k_;
    d = dParam_;
    n = n_;
    type = type_;

}
ImproperCVFF::ImproperCVFF(double k_, int d_, int n_, int type_) {
    for (int i=0; i<4; i++) {
        ids[i] = -1;
    }
    k = k_;
    d = d_;
    n = n_;
    type = type_;

}

ImproperCVFFType::ImproperCVFFType(ImproperCVFF *imp) {
    k = imp->k;
    d = imp->d;
    n = imp->n;
}
bool ImproperCVFFType::operator==(const ImproperCVFFType &other) const {
    return k == other.k and d == other.d and n == other.n;
}

std::string ImproperCVFF::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' d='" << d << "' n='" << n << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID\
_c='" << ids[2] << "' atomID_d='" << ids[3] << "'/>\n";
  return ss.str();
}

std::string ImproperCVFFType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' d='" << d << "' n='" << n;
  return ss.str();
}



void export_Impropers() {
    py::class_<ImproperHarmonic, SHARED(ImproperHarmonic)> ( "SimImproperHarmonic", py::init<>())
        .def_readwrite("type", &ImproperHarmonic::type)
        .def_readonly("thetaEq", &ImproperHarmonic::thetaEq)
        .def_readonly("k", &ImproperHarmonic::k)
        .def_readonly("ids", &ImproperHarmonic::ids)

    ;
    py::class_<ImproperCVFF, SHARED(ImproperCVFF)> ( "SimImproperCVFF", py::init<>())
        .def_readwrite("type", &ImproperCVFF::type)
        .def_readonly("k", &ImproperCVFF::k)
        .def_readonly("d", &ImproperCVFF::d)
        .def_readonly("n", &ImproperCVFF::n)
        .def_readonly("ids", &ImproperCVFF::ids)

    ;

}
