#include "Units.h"
#include "boost_for_export.h"
namespace py = boost::python;

void Units::setLJ() {
    boltz = 1;
    hbar  = 1;
    mvv_to_eng = 1;
    qqr_to_eng = 1;
    nktv_to_press = 1;
    ftm_to_v = 1.0;
    if (unitType != UNITS::LJ) {
        *dt = 0.005;
    }
    unitType = UNITS::LJ;
    toSIDensity = 1;
}

void Units::setReal() {
    //kcal, ang, femptoseconds
    boltz = 0.0019872067;
    hbar   = 15.178728314015; // kcal/mol * fs
    mvv_to_eng = 48.88821291 * 48.88821291;
    nktv_to_press = 68568.415;
    qqr_to_eng = 332.06371;
    ftm_to_v = 1.0f / (48.88821291 * 48.88821291);
    if (unitType != UNITS::REAL) {
        *dt = 1.0;
    }
    unitType = UNITS::REAL;
    toSIDensity = 1.0 / (1000*6.02214e23*1e-30);
}


void export_Units() {
    py::class_<Units, boost::noncopyable> (
        "Units",
        py::no_init
    )
    //.def("populateOnGrid", &InitializeAtoms::populateOnGrid,
    //        (py::arg("bounds"),
    //         py::arg("handle"),
    //         py::arg("n"))
    //    )
    //.staticmethod("populateOnGrid")
    .def_readonly("toSIDensity", &Units::toSIDensity)
    .def("setReal", &Units::setReal)
    .def("setLJ", &Units::setLJ)
    ;
}
