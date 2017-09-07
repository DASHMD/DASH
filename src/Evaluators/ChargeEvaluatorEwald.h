#pragma once
#ifndef EVALUATOR_EWALD
#define EVALUATOR_EWALD

#include "cutils_math.h"

class ChargeEvaluatorEwald {
    public:
        float alpha;
        float qqr_to_eng;
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            if (lenSqr < 1e-10) {
                lenSqr = 1e-10;
            }
            float r2inv = 1.0f/lenSqr;
            float rinv = sqrtf(r2inv);
            float len = sqrtf(lenSqr);
            float forceScalar = qqr_to_eng * qi*qj*(erfcf((alpha*len))*rinv+(2.0f*0.5641895835477563f*alpha)*exp(-alpha*alpha*lenSqr));
            if (multiplier < 1.0f) {
                float correctionVal = qqr_to_eng * qi * qj * rinv;
                forceScalar -= (1.0f - multiplier) * correctionVal;
            }

            forceScalar *= r2inv;
            return dr * forceScalar;
        }
        inline __device__ float energy(float lenSqr, float qi, float qj, float multiplier) {
            if (lenSqr < 1e-10) {
                lenSqr = 1e-10;
            }
            float len=sqrtf(lenSqr);
            float rinv = 1.0f/len;                 
            float prefactor = qqr_to_eng * qi * qj * rinv;
            float eng = prefactor * erfcf(alpha*len);
            if (multiplier < 1.0f) {
                eng -= (1 - multiplier) * prefactor;
            }
            return 0.5f * eng;

        }
        ChargeEvaluatorEwald(float alpha_, float qqr_to_eng_) : alpha(alpha_), qqr_to_eng(qqr_to_eng_) {};

};

#endif
