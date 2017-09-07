#pragma once
#ifndef FIXBONDQUARTIC_H
#define FIXBONDQUARTIC_H

#include "Bond.h"
#include "FixBond.h"
#include "BondEvaluatorQuartic.h"
void export_FixBondQuartic();

class FixBondQuartic : public FixBond<BondQuartic, BondGPU, BondQuarticType> {

public:
    //int maxBondsPerBlock;
    //DataSet *eng;
    //DataSet *press;

    FixBondQuartic(boost::shared_ptr<State> state_, std::string handle);

    ~FixBondQuartic(){};

    void compute(int);
    void singlePointEng(float *);
    std::string restartChunk(std::string format);
    bool readFromRestart();
    BondEvaluatorQuartic evaluator;

    // HEY - NEED TO IMPLEMENT REFRESHATOMS
    // consider that if you do so, max bonds per block could change
    //bool refreshAtoms();

    void createBond(Atom *, Atom *, double, double, double, double, int);  // by ids
    void setBondTypeCoefs(int, double, double, double, double);

    const BondQuartic getBond(size_t i) {
        return boost::get<BondQuartic>(bonds[i]);
    }
    virtual std::vector<BondVariant> *getBonds() {
        return &bonds;
    }

    //std::vector<pair<int, std::vector<int> > > neighborlistExclusions();

};

#endif
