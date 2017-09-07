#include "FixPressureBerendsen.h"
#include "State.h"
#include "Mod.h"
namespace py = boost::python;
const std::string BerendsenType = "Langevin";
using namespace MD_ENGINE;

FixPressureBerendsen::FixPressureBerendsen(boost::shared_ptr<State> state_, std::string handle_, double pressure_, double period_, int applyEvery_) : Interpolator(pressure_), Fix(state_, handle_, "all", BerendsenType, false, true, false, applyEvery_), pressureComputer(state, "scalar"), period(period_) {
    bulkModulus = 10; //lammps
    maxDilation = 0.00001;
    requiresPerAtomVirials=true;
};

bool FixPressureBerendsen::prepareForRun() {
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    pressureComputer.prepareForRun(); 
    return true;
}

bool FixPressureBerendsen::stepFinal() {
    double dilationUpper = 1.0 + maxDilation;
    double dilationLower = 1.0 - maxDilation;
    pressureComputer.computeScalar_GPU(true, 1);
    computeCurrentVal(state->turn);
    double target = getCurrentVal();
    cudaDeviceSynchronize();
    pressureComputer.computeScalar_CPU();
    double pressure = pressureComputer.pressureScalar;
    double dilation = std::pow(1.0 - state->dt/period * (target - pressure) / bulkModulus, 1.0/3.0);
    if (dilation < dilationLower) {
        dilation = dilationLower;
    } else if (dilation > dilationUpper) {
        dilation = dilationUpper;
    }
    Mod::scaleSystem(state, make_float3(dilation, dilation, dilation));
    return true;
}

bool FixPressureBerendsen::postRun() {
    finished = true;
    return true;
}

void FixPressureBerendsen::setParameters(double maxDilation_) {
    maxDilation = maxDilation_;
}

void export_FixPressureBerendsen() {
    py::class_<FixPressureBerendsen, boost::shared_ptr<FixPressureBerendsen>, py::bases<Fix> > (
        "FixPressureBerendsen", 
        py::init<boost::shared_ptr<State>, std::string, double, double, int>(
            py::args("state", "handle", "pressure", "period", "applyEvery")
            )
    )
    .def("setParameters", &FixPressureBerendsen::setParameters, (py::arg("maxDilation")=-1))
    ;
}
