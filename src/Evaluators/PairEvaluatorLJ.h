#pragma once
#ifndef EVALUATOR_LJ
#define EVALUATOR_LJ

#include "cutils_math.h"

class EvaluatorLJ {
    public:
        inline __device__ float3 force(float3 dr, float params[3], float lenSqr, float multiplier) {
            if (multiplier) {
                float epstimes24 = params[1];
                float sig6 = params[2];
                float p1 = epstimes24*2*sig6*sig6;
                float p2 = epstimes24*sig6;
                float r2inv = 1/lenSqr;
                float r6inv = r2inv*r2inv*r2inv;
                float forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;
                return dr * forceScalar;
            }
            return make_float3(0, 0, 0);
        }
        
        // double precision version
        inline __device__ double3 force(double3 dr, double params[3], double lenSqr, double multiplier) {
            if (multiplier) {
                double epstimes24 = params[1];
                double sig6 = params[2];
                double p1 = epstimes24*2*sig6*sig6;
                double p2 = epstimes24*sig6;
                double r2inv = 1/lenSqr;
                double r6inv = r2inv*r2inv*r2inv;
                double forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;
                return dr * forceScalar;
            }
            return make_double3(0, 0, 0);
        }


        inline __device__ float energy(float params[3], float lenSqr, float multiplier) {
            if (multiplier) {
                float eps = params[1] / 24.0f;
                float sig6 = params[2];
                float r2inv = 1/lenSqr;
                float r6inv = r2inv*r2inv*r2inv;
                float sig6r6inv = sig6 * r6inv;
                float rCutSqr = params[0];
                float rCut6 = rCutSqr*rCutSqr*rCutSqr;

                float sig6InvRCut6 = sig6 / rCut6;
                float offsetOver4Eps = sig6InvRCut6*(sig6InvRCut6-1.0f);
                return 0.5f * 4*eps*(sig6r6inv*(sig6r6inv-1.0f) - offsetOver4Eps) * multiplier; //0.5 b/c we need to half-count energy b/c pairs are redundant
            }
            return 0;
        }

};

#endif
