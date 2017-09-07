#pragma once
#ifndef FIXBONDFENE_H
#define FIXBONDFENE_H

#include "Bond.h"
#include "FixBond.h"
#include "BondEvaluatorFENE.h"
void export_FixBondFENE();

class FixBondFENE : public FixBond<BondFENE, BondGPU, BondFENEType> {

public:
    //int maxBondsPerBlock;
    //DataSet *eng;
    //DataSet *press;

    FixBondFENE(boost::shared_ptr<State> state_, std::string handle);

    ~FixBondFENE(){};

    void compute(int);
    void singlePointEng(float *);
    std::string restartChunk(std::string format);
    bool readFromRestart();
    BondEvaluatorFENE evaluator;

    // HEY - NEED TO IMPLEMENT REFRESHATOMS
    // consider that if you do so, max bonds per block could change
    //bool refreshAtoms();

    void createBond(Atom *, Atom *, double, double, double, double, int);  // by ids
    void setBondTypeCoefs(int, double, double, double, double);

    const BondFENE getBond(size_t i) {
        return boost::get<BondFENE>(bonds[i]);
    }
    virtual std::vector<BondVariant> *getBonds() {
        return &bonds;
    }

    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();

};

#endif
