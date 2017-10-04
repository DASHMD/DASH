#pragma once

#include "cutils_math.h"

class EvaluatorNone {
    public:
        char x; //variables on device must have non-zero size;
        inline __device__ float3 force(float3 dr, float params[1], float lenSqr, float multiplier) {
            return make_float3(0, 0, 0);
        }
        inline __device__ double3 force(double3 dr, double params[1], double lenSqr, double multiplier) {
            return make_double3(0, 0, 0);
        }
        inline __device__ float energy(float params[0], float lenSqr, float multiplier) {
            return 0;
        }

};

