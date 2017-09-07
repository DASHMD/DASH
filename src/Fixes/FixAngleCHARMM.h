#pragma once
#ifndef FIXANGLECHARMM_H
#define FIXANGLECHARMM_H

#include "FixPotentialMultiAtom.h"
#include "Angle.h"
#include "AngleEvaluatorCHARMM.h"

void export_FixAngleCHARMM();

class FixAngleCHARMM : public FixPotentialMultiAtom<AngleVariant, AngleCHARMM, Angle, AngleGPU, AngleCHARMMType, 3> {

private:
    AngleEvaluatorCHARMM evaluator; 
public:
    //DataSet *eng;
    //DataSet *press;

    FixAngleCHARMM(boost::shared_ptr<State> state_, std::string handle);

    void compute(int);
    void singlePointEng(float *);

    void createAngle(Atom *, Atom *, Atom *, double, double, double, double, int type_);
    void setAngleTypeCoefs(double, double, double, double, int);

    bool readFromRestart();

};

#endif
