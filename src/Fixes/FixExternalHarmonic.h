#pragma once

#include "FixExternal.h"
#include "ExternalEvaluatorHarmonic.h"



void export_FixExternalHarmonic();

class FixExternalHarmonic : public FixExternal {
	public:
		FixExternalHarmonic(SHARED(State), std::string handle_, std::string groupHandle_,
							Vector k_, Vector r0_);
	    float3 k;    // component-wise coefficent from harmonic potential
	    float3 r0;   // origin for harmonic potential
	    	
        void compute(int);

	    bool prepareForRun();
	    	
        bool postRun();

        void singlePointEng(float *);

	    EvaluatorExternalHarmonic evaluator; // evaluator for harmonic wall interactions

};




