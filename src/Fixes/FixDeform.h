#pragma once
#include "Fix.h"
#include "Interpolator.h"
namespace py = boost::python;
void export_FixDeform();
class FixDeform : public Fix{
public:
    FixDeform(boost::shared_ptr<State> state_, std::string handle_,
            std::string groupHandle_, double deformRate_, Vector multipler_=Vector(1, 1, 1), int applyEvery_=1);

    FixDeform(boost::shared_ptr<State> state_, std::string handle_,
            std::string groupHandle_, py::object deformRateFunc_, Vector multipler_=Vector(1, 1, 1), int applyEvery_=1);

    FixDeform(boost::shared_ptr<State> state_, std::string handle_,
            std::string groupHandle_, py::list intervals_, py::list rates_, Vector multipler_=Vector(1, 1, 1), int applyEvery_=1);
    Interpolator deformRateInterpolator;
    Vector multiplier;
    bool prepareForRun();
    bool stepFinal();
    //sets deform fix to linearly move to set volume over next run
    void toVolume(double vol);
    double setPtVolume;


};
