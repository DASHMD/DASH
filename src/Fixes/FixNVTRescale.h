#pragma once
#ifndef FIXNVTRESCALE_H
#define FIXNVTRESCALE_H

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
class Bounds;
class State;

void export_FixNVTRescale();
class FixNVTRescale : public Interpolator, public Fix {

private:
    int curIdx;

    BoundsGPU boundsGPU;

    MD_ENGINE::DataComputerTemperature tempComputer;
    bool prepareFinal();
    //void compute(int);
    bool postRun();
    bool stepFinal();

public:
    boost::shared_ptr<Bounds> thermoBounds;
    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, boost::python::list intervals, boost::python::list temps_, int applyEvery = 10, int orderPreference = 35);
    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, boost::python::object tempFunc_, int applyEvery = 10, int orderPreference = 35);
    FixNVTRescale(boost::shared_ptr<State>, std::string handle_, std::string groupHandle_, double temp_, int applyEvery = 10, int orderPreference = 35);
    Interpolator *getInterpolator(std::string);

};

#endif
