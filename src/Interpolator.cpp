#include "Interpolator.h"
#include "Logging.h"
enum thermoType {interval, constant, pyFunc};
namespace py = boost::python;
Interpolator::Interpolator(py::list intervals_, py::list vals_) {
    mode = thermoType::interval;
    int len = boost::python::len(intervals_);
    for (int i=0; i<len; i++) {
        boost::python::extract<double> intPy(intervals_[i]);
        boost::python::extract<double> valPy(vals_[i]);
        if (!intPy.check() or !valPy.check()) {
            assert(intPy.check() and valPy.check());
        }
        double interval = intPy;
        double val = valPy;
        intervals.push_back(interval);
        vals.push_back(val);
    }
    curIntervalIdx = 0;
    finished = false;
    mdAssert(intervals[0] == 0 and intervals.back() == 1, "Invalid intervals given to interpolator");
}
Interpolator::Interpolator(double val_) {
    mode = thermoType::constant;
    constVal = val_;
    //mdAssert(constVal > 0, "Invalid value given to interpolator");
}
Interpolator::Interpolator(py::object valFunc_) {
    mode = thermoType::pyFunc;
    valFunc = valFunc_;
    mdAssert(PyCallable_Check(valFunc.ptr()), "Must give callable function to interpolator");

}


void Interpolator::computeCurrentVal(int64_t turn) {
    if (mode == thermoType::interval) {
        if (finished) {
            currentVal = vals.back();
        } else {
            double frac = (turn-turnBeginRun) / (double) (turnFinishRun - turnBeginRun);
            while (frac > intervals[curIntervalIdx+1] and curIntervalIdx < intervals.size()-1) {
                curIntervalIdx++;
            }
            double valA = vals[curIntervalIdx];
            double valB = vals[curIntervalIdx+1];
            double intA = intervals[curIntervalIdx];
            double intB = intervals[curIntervalIdx+1];
            double fracThroughInterval = (frac-intA) / (intB-intA);
            currentVal = valB*fracThroughInterval + valA*(1-fracThroughInterval);
        }
    } else if (mode == thermoType::constant) {
        currentVal = constVal;
    } else if (mode == thermoType::pyFunc) {
        currentVal = py::call<double>(valFunc.ptr(), turnBeginRun, turnFinishRun, turn);
    }
}

double Interpolator::getCurrentVal() {
    return currentVal;
}

void Interpolator::finishRun() {
    finished = true;
}
