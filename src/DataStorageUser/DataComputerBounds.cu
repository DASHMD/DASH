#include "DataComputerBounds.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerBounds::DataComputerBounds(State *state_) : DataComputer(state_, "scalar", false) {
}


void DataComputerBounds::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    storedBounds = Bounds(state->boundsGPU);
}





void DataComputerBounds::computeScalar_CPU() {
}



void DataComputerBounds::appendScalar(boost::python::list &vals) {
    vals.append(storedBounds);
}

void DataComputerBounds::prepareForRun() {
}

