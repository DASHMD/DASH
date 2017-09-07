#pragma once
#ifndef FIXANGLECOSINEDELTA_H
#define FIXANGLECOSINEDELTA_H

#include "FixPotentialMultiAtom.h"
#include "Angle.h"
#include "AngleEvaluatorCosineDelta.h"

void export_FixAngleCosineDelta();

class FixAngleCosineDelta : public FixPotentialMultiAtom<AngleVariant, AngleCosineDelta, Angle, AngleGPU, AngleCosineDeltaType, 3> {

private:
    AngleEvaluatorCosineDelta evaluator; 
public:
    //DataSet *eng;
    //DataSet *press;

    FixAngleCosineDelta(boost::shared_ptr<State> state_, std::string handle);

    void compute(int);
    void singlePointEng(float *);

    void createAngle(Atom *, Atom *, Atom *, double, double, int type_);
    void setAngleTypeCoefs(int, double, double);

    bool readFromRestart();

};

#endif
