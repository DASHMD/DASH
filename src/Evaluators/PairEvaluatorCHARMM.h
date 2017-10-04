#pragma once

#include "cutils_math.h"

class EvaluatorCHARMM {
    public:
        inline __device__ float3 force(float3 dr, float params[5], float lenSqr, float multiplier) {
            if (multiplier) {
                bool isNorm = multiplier != mult14;
                float epstimes24 = isNorm ? params[1] : params[3];
                float sig6 = isNorm ? params[2] : params[4];
                float p1 = epstimes24*2*sig6*sig6;
                float p2 = epstimes24*sig6;
                float r2inv = 1/lenSqr;
                float r6inv = r2inv*r2inv*r2inv;
                float forceScalar = r6inv * r2inv * (p1 * r6inv - p2) * multiplier;
                return dr * forceScalar;
            }
            return make_float3(0, 0, 0);
        }

        // double precision
        inline __device__ double3 force(double3 dr, double params[5], double lenSqr, double multiplier) {
            if (multiplier) {
                bool isNorm = multiplier != mult14;
                double epstimes24 = isNorm ? params[1] : params[3];
                double sig6 = isNorm ? params[2] : params[4];
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
                bool isNorm = multiplier != mult14;
                float eps = (isNorm ? params[1] : params[3]) / 24.0f;
                float sig6 = isNorm ? params[2] : params[4];
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
        float mult14;
        EvaluatorCHARMM(float mult14_) {
            mult14 = mult14_;
        }

};

