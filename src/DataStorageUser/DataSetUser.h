#pragma once
#ifndef DATASETUSER_H
#define DATASETUSER_H
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include "Python.h"
#include <boost/shared_ptr.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <string.h>
class State;
void export_DataSetUser();
namespace MD_ENGINE {

class DataComputer;
enum COMPUTEMODE {INTERVAL, PYTHON};
enum DATAMODE {SCALAR, VECTOR, TENSOR};
enum DATATYPE {TEMPERATURE, PRESSURE, ENERGY, BOUNDS, DIPOLARCOUPLING};
class DataSetUser {
private:
    State *state;
public:
    DataSetUser(State *, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, int);
    DataSetUser(State *, boost::shared_ptr<DataComputer> computer_, uint32_t groupTag_, boost::python::object);

    boost::python::list turns;
    boost::python::list vals;

    uint32_t groupTag;
    boost::shared_ptr<DataComputer> computer;
    int computeMode;
    int64_t nextCompute;

    bool requiresVirials();
    bool requiresPerAtomVirials();

    void prepareForRun();
    void computeData();
    void appendData();

    void setPyFunc(boost::python::object func_);
    boost::python::object getPyFunc();
    boost::python::object pyFunc;
    PyObject *pyFuncRaw;
    int interval;
    int64_t setNextTurn(int64_t currentTurn); //called externally 
};

}



#endif
