#pragma once

#include "cutils_math.h"

class EvaluatorEField {
    public:
        float angToBohr;
        //not a force - just vector representing the electric field
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            float3 inBohr = dr * angToBohr;
            float3 field = qj * inBohr / powf(lenSqr, 1.5);
            return field;
        }
        inline __device__ float energy(float lenSqr, float qi, float qj, float multiplier) {
            return 0;
        }
        EvaluatorEField() : angToBohr(1.88977161646) {};

};

