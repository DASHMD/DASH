#pragma once
#ifndef FIXDIHEDRALOPLS_H
#define FIXDIHEDRALOPLS_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>

#include "FixPotentialMultiAtom.h"
#include "Dihedral.h"
#include "DihedralEvaluatorOPLS.h"

void export_FixDihedralOPLS();

class FixDihedralOPLS : public FixPotentialMultiAtom<DihedralVariant, DihedralOPLS, Dihedral, DihedralGPU, DihedralOPLSType, 4> {

private:
    DihedralEvaluatorOPLS evaluator;
public:
    //DataSet *eng;
    //DataSet *press;

    FixDihedralOPLS(boost::shared_ptr<State> state_, std::string handle);

    void compute(int);
    void singlePointEng(float *);

    void createDihedral(Atom *, Atom *, Atom *, Atom *, double, double, double, double, int);
    void createDihedralPy(Atom *, Atom *, Atom *, Atom *, boost::python::list, int);
    void setDihedralTypeCoefs(int, boost::python::list);

    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();
    bool readFromRestart();

};

#endif
