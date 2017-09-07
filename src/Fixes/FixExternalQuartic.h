#pragma once

#include "FixExternal.h"
#include "ExternalEvaluatorQuartic.h"



void export_FixExternalQuartic();

class FixExternalQuartic : public FixExternal {
	public:
		FixExternalQuartic(SHARED(State), std::string handle_, std::string groupHandle_,
							Vector k1_, Vector k2_, Vector k3_, Vector k4, Vector r0_);
	    float3 k1;    // component-wise coefficent from linear potential
	    float3 k2;    // component-wise coefficent from harmonic potential
	    float3 k3;    // component-wise coefficent from cubic potential
	    float3 k4;    // component-wise coefficent from quartic potential
	    float3 r0;    // origin for potential
		
        void compute(int);

	    bool prepareForRun();
		
        bool postRun();

        void singlePointEng(float *);

	    EvaluatorExternalQuartic evaluator; // evaluator for harmonic wall interactions

};




