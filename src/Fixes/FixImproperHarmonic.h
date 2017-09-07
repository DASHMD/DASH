#pragma once
#ifndef FIXIMPROPERHARMONIC_H
#define FIXIMPROPERHARMONIC_H

#include "FixPotentialMultiAtom.h"
#include "Improper.h"
#include "ImproperEvaluatorHarmonic.h"
void export_FixImproperHarmonic();

class FixImproperHarmonic: public FixPotentialMultiAtom<ImproperVariant, ImproperHarmonic, Improper, ImproperGPU, ImproperHarmonicType, 4> {

    public:
        //DataSet *eng;
        //DataSet *press;

        FixImproperHarmonic(SHARED(State) state_, std::string handle);

        void compute(int);
        void singlePointEng(float *);
        bool readFromRestart();

        void createImproper(Atom *, Atom *, Atom *, Atom *, double, double, int);
        void setImproperTypeCoefs(int, double, double);
        ImproperEvaluatorHarmonic evaluator;
        //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();

};

#endif
