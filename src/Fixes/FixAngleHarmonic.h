#pragma once
#ifndef FIXANGLEHARMONIC_H
#define FIXANGLEHARMONIC_H

#include "FixPotentialMultiAtom.h"
#include "Angle.h"
#include "AngleEvaluatorHarmonic.h"

void export_FixAngleHarmonic();

class FixAngleHarmonic : public FixPotentialMultiAtom<AngleVariant, AngleHarmonic, Angle, AngleGPU, AngleHarmonicType, 3> {

private:
    AngleEvaluatorHarmonic evaluator; 
public:
    //DataSet *eng;
    //DataSet *press;

    FixAngleHarmonic(boost::shared_ptr<State> state_, std::string handle);

    void compute(int);
    void singlePointEng(float *);

    void createAngle(Atom *, Atom *, Atom *, double, double, int type_);
    void setAngleTypeCoefs(int, double, double);

    bool readFromRestart();

};

#endif
