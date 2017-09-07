#pragma once
#ifndef FIXPRESSUREBERENDSEN
#define FIXPRESSUREBERENDSEN
#include "Fix.h"
#include "Interpolator.h"
#include "DataComputerPressure.h"
void export_FixPressureBerendsen();

namespace MD_ENGINE {
    class FixPressureBerendsen : public Interpolator, public Fix {
    public: 
        FixPressureBerendsen(boost::shared_ptr<State> state_, std::string handle_, double pressure_, double period_, int applyEvery_);

        bool prepareForRun();
        bool stepFinal();
        bool postRun();
        DataComputerPressure pressureComputer;
        double period;
        double bulkModulus;
        double maxDilation;

        void setParameters(double maxDilation_);

    };
};
#endif
