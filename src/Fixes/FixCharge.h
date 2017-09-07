#pragma once
#ifndef FIX_CHARGE_H
#define FIX_CHARGE_H

//#include "AtomParams.h"
#include "GPUArrayTex.h"
#include "Fix.h"

class State;

void export_FixCharge();

class FixCharge : public Fix {

public:
    FixCharge(boost::shared_ptr<State> state_,
              std::string handle_, std::string groupHandle_,
              std::string type_, bool forceSingle_);

    bool prepareForRun();
    virtual void compute(int) { };

};

#endif
