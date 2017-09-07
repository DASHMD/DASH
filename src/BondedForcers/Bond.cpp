#include "Bond.h"
#include "boost_for_export.h"

namespace py = boost::python;

//BondHarmonicType::BondHarmonicType(BondHarmonic *bond) {
//    k = bond->k;
//    r0 = bond->r0;
//}

bool BondHarmonicType::operator==(const BondHarmonicType &other) const {
    return k == other.k and r0 == other.r0;
}





BondHarmonic::BondHarmonic(Atom *a, Atom *b, double k_, double r0_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    k = k_;
    r0 = r0_;
    type = type_;
}
BondHarmonic::BondHarmonic(double k_, double r0_, int type_) {
    k = k_;
    r0 = r0_;
    type = type_;
}

void BondGPU::takeIds(Bond *b) { 
    myId = b->ids[0];
    otherId = b->ids[1];
}

std::string BondHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' r0='" << r0;
  return ss.str();
}

std::string BondHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' r0='" << r0 << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'/>\n";
  return ss.str();
}

void export_BondHarmonic() {
  
    boost::python::class_<BondHarmonic,SHARED(BondHarmonic)> ( "BondHarmonic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondHarmonic::ids)
        .def_readwrite("k", &BondHarmonic::k)
        .def_readwrite("r0", &BondHarmonic::r0)
    ;
}

//BondQuarticType::BondQuarticType(BondQuartic *bond) {
//    k2 = bond->k2;
//    k3 = bond->k3;
//    k4 = bond->k4;
//    r0 = bond->r0;
//}

bool BondQuarticType::operator==(const BondQuarticType &other) const {
    return k2 == other.k2 and k3 == other.k3 and k4 == other.k4 and r0 == other.r0;
}

BondQuartic::BondQuartic(Atom *a, Atom *b, double k2_, double k3_, double k4_, double r0_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    k2 = k2_;
    k3 = k3_;
    k4 = k4_;
    r0 = r0_;
    type = type_;
}
BondQuartic::BondQuartic(double k2_, double k3_, double k4_, double r0_, int type_) {
    k2 = k2_;
    k3 = k3_;
    k4 = k4_;
    r0 = r0_;
    type = type_;
}

std::string BondQuarticType::getInfoString() {
  std::stringstream ss;
  ss << " k2='" << k2 << " k3='" << k3 << " k4='" << k4 << "' r0='" << r0;
  return ss.str();
}

std::string BondQuartic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << " k2='" << k2 << " k3='" << k3 << " k4='" << k4 << "' r0='" << r0 << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'/>\n";
  return ss.str();
}

void export_BondQuartic() {
  
    boost::python::class_<BondQuartic,SHARED(BondQuartic)> ( "BondQuartic", boost::python::init<>())
//         .def(boost::python::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondQuartic::ids)
        .def_readwrite("k2", &BondQuartic::k2)
        .def_readwrite("k3", &BondQuartic::k3)
        .def_readwrite("k4", &BondQuartic::k4)
        .def_readwrite("r0", &BondQuartic::r0)
    ;
}








//bond FENE
bool BondFENEType::operator==(const BondFENEType &other) const {
    return k == other.k and r0 == other.r0 and eps == other.eps and sig == other.sig;
}





BondFENE::BondFENE(Atom *a, Atom *b, double k_, double r0_, double eps_, double sig_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    k = k_;
    r0 = r0_;
    eps = eps_;
    sig = sig_;
    type = type_;
}
BondFENE::BondFENE(double k_, double r0_, double eps_, double sig_, int type_) {
    k = k_;
    r0 = r0_;
    eps = eps_;
    sig = sig_;
    type = type_;
}

std::string BondFENEType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' r0='" << r0 << "' eps='" << eps << "' sig='" << sig;
  return ss.str();
}

std::string BondFENE::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' r0='" << r0 << "' eps='" << eps << "' sig='" << sig << "' atomID_a='" << ids[0] <<  "' atomID_b='" << ids[1] << "'/>\n";
  return ss.str();
}

void export_BondFENE() {
  
    py::class_<BondFENE,SHARED(BondFENE)> ( "BondFENE", py::init<>())
//         .def(py::init<int, int ,double, double,int>())
        .def_readonly("ids", &BondFENE::ids)
        .def_readwrite("k", &BondFENE::k)
        .def_readwrite("r0", &BondFENE::r0)
        .def_readwrite("eps", &BondFENE::eps)
        .def_readwrite("sig", &BondFENE::sig)
    ;
}
