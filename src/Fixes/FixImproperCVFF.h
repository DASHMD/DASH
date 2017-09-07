#pragma once
#ifndef FIXIMPROPERCVFF_H
#define FIXIMPROPERCVFF_H

#include "FixPotentialMultiAtom.h"
#include "Improper.h"
#include "ImproperEvaluatorCVFF.h"
void export_FixImproperCVFF();

class FixImproperCVFF: public FixPotentialMultiAtom<ImproperVariant, ImproperCVFF, Improper, ImproperGPU, ImproperCVFFType, 4> {

    public:

        FixImproperCVFF(SHARED(State) state_, std::string handle);

        void compute(int);
        void singlePointEng(float *);
        bool readFromRestart();

        void createImproper(Atom *, Atom *, Atom *, Atom *, double, int, int, int);
        void setImproperTypeCoefs(int, double, int, int);
        ImproperEvaluatorCVFF evaluator;

};

#endif
