#pragma once

#include "cutils_math.h"

class ChargeEvaluatorDSF {
    public:
        float alpha;
        float A;
        float shift;
        float qqr_to_eng;
        float r_cut;
        inline __device__ float3 force(float3 dr, float lenSqr, float qi, float qj, float multiplier) {
            float r2inv = 1.0f/lenSqr;
            float rinv = sqrtf(r2inv);
            float len = sqrtf(lenSqr);
            float forceScalar = qqr_to_eng * qi*qj*(erfcf((alpha*len))*r2inv+A*expf(-alpha*alpha*lenSqr)*rinv-shift)*rinv * multiplier;
            return dr * forceScalar;
        }


        inline __device__ double3 force(double3 dr, double lenSqr, double qi, double qj, double multiplier) {
            double r2inv = 1.0f/lenSqr;
            double rinv = sqrtf(r2inv);
            double len = sqrtf(lenSqr);
            double forceScalar = double(qqr_to_eng) * qi*qj*(erfc((double(alpha)*len))*r2inv+double(A)*exp(-double(alpha)*double(alpha)*lenSqr)*rinv-double(shift))*rinv * multiplier;
            return dr * forceScalar;
        }



        inline __device__ float energy(float lenSqr, float qi, float qj, float multiplier) {
            float r2inv = 1.0f/lenSqr;
            float rinv = sqrtf(r2inv);
            float len = sqrtf(lenSqr);
            float eng = qqr_to_eng * qi*qj*(erfcf(alpha*len)*rinv-erfcf(alpha*r_cut)/r_cut+shift*(len-r_cut))* multiplier;
            return eng*0.5;
        }
        ChargeEvaluatorDSF(float alpha_, float A_, float shift_, float qqr_to_eng_,float r_cut_) : alpha(alpha_), A(A_), shift(shift_), qqr_to_eng(qqr_to_eng_),r_cut(r_cut_) {};

};

