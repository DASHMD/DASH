#pragma once
#ifndef EVALUATOR_WALL_HARMONIC
#define EVALUATOR_WALL_HARMONIC

#include "cutils_math.h"

class EvaluatorWallHarmonic {
	public:
		float k;
        float r0;

        // default constructor
        EvaluatorWallHarmonic () {};
        EvaluatorWallHarmonic (float k_, float r0_) {
            k = k_;
            r0= r0_;
        };
       
        // setParameters method, called in FixWallHarmonic_temp::prepareForRun()
        //void setParameters(float k_, float r0_) {
         //   k = k_;
         //   r0= r0_;
       // };

        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __device__ float3 force(float magProj, float3 forceDir) {
           if (magProj < r0) { 
                float forceScalar = k * (r0 - magProj); 
                return forceDir * forceScalar;
           } else {
                return forceDir * 0.0;
           };

        };
};
#endif

