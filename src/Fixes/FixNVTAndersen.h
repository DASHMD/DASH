#pragma once
#ifndef FIXNVTANDERSEN_H
#define FIXNVTANDERSEN_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python/list.hpp>
#include <string>
#include <vector>

#include "BoundsGPU.h"
#include "Fix.h"
#include "globalDefs.h"
#include "GPUArrayDeviceGlobal.h"
#include "Interpolator.h"
#include "DataComputerTemperature.h"
#include <curand_kernel.h>

class Bounds;
class State;

void export_FixNVTAndersen();
class FixNVTAndersen : public Interpolator, public Fix {

private:
    int   seed;
    float nudt;
    void setDefaults();
    GPUArrayDeviceGlobal<curandState_t> randStates;
    BoundsGPU boundsGPU;

    MD_ENGINE::DataComputerTemperature tempComputer;
    bool prepareForRun();
    void compute(int);
    bool postRun();

public:
    boost::shared_ptr<Bounds> thermoBounds;
    FixNVTAndersen(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, boost::python::list intervals, boost::python::list temps_, float nu_= 0.01 , int applyEvery = 10);
    FixNVTAndersen(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, boost::python::object tempFunc_, float nu_ = 0.01 , int applyEvery = 10);
    FixNVTAndersen(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, double temp_, float nu_ = 0.01 , int applyEvery = 10);
    Interpolator *getInterpolator(std::string);
    void setParams(double seed);

};

#endif
