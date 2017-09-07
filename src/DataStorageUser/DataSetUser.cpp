#include "DataSetUser.h"
#include "Logging.h"
#include "State.h"
#include "DataComputer.h"

namespace py = boost::python;
using namespace MD_ENGINE;

DataSetUser::DataSetUser(State *state_, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, boost::python::object pyFunc_) : state(state_), computeMode(COMPUTEMODE::PYTHON), groupTag(groupTag_), computer(computer_), pyFunc(pyFunc_), pyFuncRaw(pyFunc_.ptr()) {
    mdAssert(PyCallable_Check(pyFuncRaw), "Non-function passed to data set");
    setNextTurn(state->turn);
}

DataSetUser::DataSetUser(State *state_, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, int interval_) : state(state_), computeMode(COMPUTEMODE::INTERVAL), groupTag(groupTag_), computer(computer_), interval(interval_) {
    nextCompute = state->turn;

}
void DataSetUser::prepareForRun() {
    computer->prepareForRun();
    
}
void DataSetUser::computeData() {
    computer->compute_GPU(true, groupTag);
    //if (dataMode == DATAMODE::SCALAR) {
    //    computer->computeScalar_GPU(true, groupTag);
    //} else if (dataMode == DATAMODE::VECTOR) {
    //    computer->computeVector_GPU(true, groupTag);
    //} else if (dataMode == DATAMODE::TENSOR) {
    //    computer->computeTensor_GPU(true, groupTag);
    //}
    turns.append(state->turn);
}

void DataSetUser::appendData() {
    computer->compute_CPU();
    computer->appendData(vals);
}

        
int64_t DataSetUser::setNextTurn(int64_t currentTurn) {
    if (computeMode == COMPUTEMODE::INTERVAL) {
        nextCompute = currentTurn + interval;
    } else {
        nextCompute = py::call<int64_t>(pyFuncRaw, currentTurn);
    }
    return nextCompute;
}

boost::python::object DataSetUser::getPyFunc() {
    return pyFunc;
}

void DataSetUser::setPyFunc(boost::python::object func_) {
    pyFunc = func_;
    pyFuncRaw = pyFunc.ptr();
    mdAssert(PyCallable_Check(pyFuncRaw), "Non-function passed to data set");
}

bool DataSetUser::requiresVirials() {
    //printf("I AM QUERIED\n");
    return computer->requiresVirials;
}
bool DataSetUser::requiresPerAtomVirials() {
    return computer->requiresPerAtomVirials;
}

void export_DataSetUser() {
    boost::python::class_<DataSetUser, boost::shared_ptr<DataSetUser>, boost::noncopyable>("DataSetUser", boost::python::no_init)
    .def_readonly("turns", &DataSetUser::turns)
    .def_readonly("vals", &DataSetUser::vals)
    .def_readwrite("interval", &DataSetUser::interval)
    .add_property("pyFunc", &DataSetUser::getPyFunc, &DataSetUser::setPyFunc);
 //   .def("getDataSet", &DataManager::getDataSet)
    ;
}
