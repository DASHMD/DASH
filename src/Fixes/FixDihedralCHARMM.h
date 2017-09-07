#pragma once
#ifndef FIXDIHEDRALCHARMM_H
#define FIXDIHEDRALCHARMM_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>

#include "FixPotentialMultiAtom.h"
#include "Dihedral.h"
#include "DihedralEvaluatorCHARMM.h"

void export_FixDihedralCHARMM();

class FixDihedralCHARMM : public FixPotentialMultiAtom<DihedralVariant, DihedralCHARMM, Dihedral, DihedralGPU, DihedralCHARMMType, 4> {

private:
    DihedralEvaluatorCHARMM evaluator;
public:
    //DataSet *eng;
    //DataSet *press;

    FixDihedralCHARMM(boost::shared_ptr<State> state_, std::string handle);

    void compute(int);
    void singlePointEng(float *);

    void createDihedral(Atom *, Atom *, Atom *, Atom *, double, int, double, int);
    void setDihedralTypeCoefs(int, double, int, double);

    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();
    bool readFromRestart();

};

#endif
