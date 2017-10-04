#pragma once
#ifndef FIX_CHARGEPAIR_DSF_H
#define FIX_CHARGEPAIR_DSF_H

//#include "AtomParams.h"
#include "GPUArrayTex.h"
#include "FixCharge.h"
#include "ChargeEvaluatorDSF.h"

class State;

void export_FixChargePairDSF();

extern const std::string chargePairDSFType;
class FixChargePairDSF : public FixCharge {

private:
    float A;  // temps for compute
    float shift;

protected:
    float alpha;
    float r_cut;

public:
    FixChargePairDSF(boost::shared_ptr<State> state_,
                     std::string handle_, std::string groupHandle_);

    bool prepareForRun();
    void setParameters(float alpha_, float r_cut_);
    void compute(int);
    void singlePointEng(float *);
    void singlePointEngGroupGroup(float *, uint32_t, uint32_t);
    ChargeEvaluatorDSF generateEvaluator();
    void setEvalWrapper();
    std::vector<float> getRCuts();

};

#endif
