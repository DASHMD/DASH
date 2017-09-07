#include "Angle.h"
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>

void Angle::takeIds(Angle *other) {
    for (int i=0; i<3; i++) {
        ids[i] = other->ids[i];
    }
}
void AngleGPU::takeIds(Angle *other) {
    for (int i=0; i<3; i++) {
        ids[i] = other->ids[i];
    }
}

//angle Harmonic
//
AngleHarmonic::AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double theta0_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    k = k_;
    theta0 = theta0_;
    type = type_;
}



AngleHarmonic::AngleHarmonic(double k_, double theta0_, int type_) {
    for (int i=0; i<3; i++) {
        ids[i] = -1;
    }
    k = k_;
    theta0 = theta0_;
    type = type_;
}



AngleHarmonicType::AngleHarmonicType(AngleHarmonic *angle) {
    k = angle->k;
    theta0 = angle->theta0;
}

std::string AngleHarmonicType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' theta0='" << theta0;
  return ss.str();
}

std::string AngleHarmonic::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' theta0='" << theta0 << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c\
='" << ids[2] << "'/>\n";
  return ss.str();
}

bool AngleHarmonicType::operator==(const AngleHarmonicType &other) const {
    return k == other.k and theta0 == other.theta0;
}
void export_AngleHarmonic() {
    boost::python::class_<AngleHarmonic, SHARED(AngleHarmonic)> ( "AngleHarmonic", boost::python::init<>())
        .def_readwrite("theta0", &AngleHarmonic::theta0)
        .def_readwrite("k", &AngleHarmonic::k)
        .def_readwrite("type", &AngleHarmonic::type)
        .def_readonly("ids", &AngleHarmonic::ids)

    ;
}

//angle CHARMM
//
AngleCHARMM::AngleCHARMM(Atom *a, Atom *b, Atom *c, double k_, double theta0_, double kub_, double rub_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    k      = k_;
    theta0 = theta0_;
    kub    = kub_;
    rub    = rub_;
    type   = type_;
}



AngleCHARMM::AngleCHARMM(double k_, double theta0_, double kub_, double rub_, int type_) {
    for (int i=0; i<3; i++) {
        ids[i] = -1;
    }
    k      = k_;
    theta0 = theta0_;
    kub    = kub_;
    rub    = rub_;
    type   = type_;
}



AngleCHARMMType::AngleCHARMMType(AngleCHARMM *angle) {
    k      = angle->k;
    theta0 = angle->theta0;
    kub    = angle->kub;
    rub    = angle->rub;
}

std::string AngleCHARMMType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' theta0='" << theta0 << " kub='" << kub << "' rub='" << rub;
  return ss.str();
}

std::string AngleCHARMM::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' theta0='" << theta0 << " kub='" << kub << "' rub='" << rub<< "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c\
='" << ids[2] << "'/>\n";
  return ss.str();
}

bool AngleCHARMMType::operator==(const AngleCHARMMType &other) const {
    return k == other.k and theta0 == other.theta0 and kub == other.kub and rub == other.rub;
}
void export_AngleCHARMM() {
    boost::python::class_<AngleCHARMM, SHARED(AngleCHARMM)> ( "AngleCHARMM", boost::python::init<>())
        .def_readwrite("theta0", &AngleCHARMM::theta0)
        .def_readwrite("k", &AngleCHARMM::k)
        .def_readwrite("rub", &AngleCHARMM::rub)
        .def_readwrite("kub", &AngleCHARMM::kub)
        .def_readwrite("type", &AngleCHARMM::type)
        .def_readonly("ids", &AngleCHARMM::ids)

    ;
}

//angle cosine delta

AngleCosineDelta::AngleCosineDelta(Atom *a, Atom *b, Atom *c, double k_, double theta0_, int type_) {
    ids[0] = a->id;
    ids[1] = b->id;
    ids[2] = c->id;
    k = k_;
    theta0 = theta0_;
    type = type_;
}



AngleCosineDelta::AngleCosineDelta(double k_, double theta0_, int type_) {
    for (int i=0; i<3; i++) {
        ids[i] = -1;
    }
    k = k_;
    theta0 = theta0_;
    type = type_;
}



AngleCosineDeltaType::AngleCosineDeltaType(AngleCosineDelta *angle) {
    k = angle->k;
    theta0 = angle->theta0;
}

std::string AngleCosineDeltaType::getInfoString() {
  std::stringstream ss;
  ss << " k='" << k << "' theta0='" << theta0;
  return ss.str();
}

std::string AngleCosineDelta::getInfoString() {
  std::stringstream ss;
  ss << "<member type='" << type << "' k='" << k << "' theta0='" << theta0 << "' atomID_a='" << ids[0] << "' atomID_b='" << ids[1] << "' atomID_c\
='" << ids[2] << "'/>\n";
  return ss.str();
}

bool AngleCosineDeltaType::operator==(const AngleCosineDeltaType &other) const {
    return k == other.k and theta0 == other.theta0;
}
void export_AngleCosineDelta() {
    boost::python::class_<AngleCosineDelta, SHARED(AngleCosineDelta)> ( "AngleCosineDelta", boost::python::init<>())
        .def_readwrite("theta0", &AngleCosineDelta::theta0)
        .def_readwrite("k", &AngleCosineDelta::k)
        .def_readwrite("type", &AngleCosineDelta::type)
        .def_readonly("ids", &AngleCosineDelta::ids)

    ;
}
