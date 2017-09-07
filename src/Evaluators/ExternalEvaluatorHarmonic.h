#pragma once

#include "cutils_math.h"

class EvaluatorExternalHarmonic {
	public:
	    float3 k;
        float3 r0;

        // default constructor
        EvaluatorExternalHarmonic () {};
        EvaluatorExternalHarmonic (float3 k_, float3 r0_ ) {
            k = k_;
            r0= r0_;
        };
       
        // force function called by compute_force_external(...) in ExternalEvaluate.h
	inline __device__ float3 force(float3 pos) {
        float3 dr = pos - r0;
        return -k * dr;
        };

        // force function called by compute_energy_external(...) in ExternalEvaluate.h
	inline __device__ float energy(float3 pos) {
        float3 dr  = pos - r0;
		float3 dr2 = dr * dr; 
        return 0.5*(k.x * dr2.x + k.y*dr2.y + k.z*dr2.z) ;
        };

};

