#pragma once
#ifndef EVALUATOR_NONE
#define EVALUATOR_NONE

#include "cutils_math.h"

class ChargeEvaluatorNone {
    public:
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            return make_float3(0, 0, 0);
        }

        inline __device__ double3 force(double3 dr, double lenSqr, double qi, double qj, double multiplier) {
            return make_double3(0, 0, 0);
        }


        inline __device__ float energy(float lenSqr, float qi, float qj, float multiplier) {
            return 0;
        }
      /*  inline __device__ float energy(float params[3], float lenSqr, float multiplier) {
            float epstimes24 = params[1];
            float sig6 = params[2];
            float r2inv = 1/lenSqr;
            float r6inv = r2inv*r2inv*r2inv;
            float sig6r6inv = sig6 * r6inv;
            return 0.5f * 4*(epstimes24 / 24)*sig6r6inv*(sig6r6inv-1.0f) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
        }*/

};

#endif
