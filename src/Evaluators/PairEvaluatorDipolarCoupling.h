#pragma once

#include "cutils_math.h"

class EvaluatorDipolarCoupling {
    public:
        //all the math with couplings, etc will be done on the CPU in double precision
        inline __device__ float3 force(float3 dr, float params[1], float lenSqr, float multiplier) {
            assert(0);
            return make_float3(0, 0, 0);
        }
        inline __device__ float energy(float params[1], float lenSqr, float multiplier) {
            return 1.0f / powf(lenSqr, 1.5);
        }

};

