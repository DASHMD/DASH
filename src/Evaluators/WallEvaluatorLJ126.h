#pragma once
#ifndef EVALUATOR_WALL_LJ126
#define EVALUATOR_WALL_LJ126

#include "cutils_math.h"

class EvaluatorWallLJ126 {
	public:
		float sigma;
        float epsilonTimes24;
        float r0;
        
        float sig2;
        float sig6;
        float sig12;
        
        // default constructor
        EvaluatorWallLJ126 () {};
        
        // another constructor here
        EvaluatorWallLJ126(float sigma_, float epsilon_, float r0_) {
            sigma = sigma_;
            epsilonTimes24 = 24.0 * epsilon_;
            r0 = r0_;
            sig2 = sigma_ * sigma_;
            sig6 = sig2 * sig2 * sig2;
            sig12 = sig6 * sig6;
        }; 
        // force function called by compute_wall_iso(...) in WallEvaluate.h
		inline __device__ float3 force(float magProj, float3 forceDir) {
            if (magProj < r0) {
                float r_inv = 1.0/magProj;
                float r2_inv = r_inv * r_inv;
                float r6_inv = r2_inv * r2_inv * r2_inv;
                float forceScalar = r6_inv * r_inv * ( ( 2.0 * epsilonTimes24 * sig12 * r6_inv - epsilonTimes24 * sig6));
                
            
                return forceDir * forceScalar ;
            } else {
                return forceDir * 0.0 ;
            };
        };
};
#endif
