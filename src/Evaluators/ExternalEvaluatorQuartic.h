#pragma once

#include "cutils_math.h"

class EvaluatorExternalQuartic {
	public:
	float3 k1;
	float3 k2;
	float3 k3;
	float3 k4;
        float3 r0;

        // default constructor
        EvaluatorExternalQuartic () {};
        EvaluatorExternalQuartic (float3 k1_, float3 k2_, float3 k3_, float3 k4_, float3 r0_ ) {
            k1 = k1_;
            k2 = k2_;
            k3 = k3_;
            k4 = k4_;
            r0 = r0_;
        };
       
        // force function called by compute_force_external(...) in ExternalEvaluate.h
	inline __device__ float3 force(float3 pos) {
                float3 dr  = pos - r0;
		float3 dr2 = dr * dr;
		float3 dr3 = dr2* dr; 
        	return -k1 - 2*k2*dr -3*k3*dr2 - 4*k4*dr3;
        };

        // force function called by compute_energy_external(...) in ExternalEvaluate.h
	inline __device__ float energy(float3 pos) {
                float3 dr  = pos - r0;
		float3 dr2 = dr * dr;
		float3 dr3 = dr2* dr; 
		float3 dr4 = dr2* dr2; 
        	return dot(k1,dr) + dot(k2,dr2) + dot(k3,dr3) + dot(k4,dr4);
        };

};

