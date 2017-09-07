#pragma once
#ifndef BONDEVALUATORHARMONIC_H
#define BONDEVALUATORHARMONIC_H

#include "Bond.h"

class BondEvaluatorHarmonic {
public:
    inline __device__ float3 force(float3 bondVec, float rSqr, BondHarmonicType bondType) {
        float r = sqrtf(rSqr);
        float dr = r - bondType.r0;
        float rk = bondType.k * dr;
        if (r > 0) {//MAKE SURE ALL THIS WORKS, I JUST BORROWED FROM LAMMPS
            float fBond = -rk/r;
            return bondVec * fBond;
        } 
        return make_float3(0, 0, 0);
    }
    inline __device__ float energy(float3 bondVec, float rSqr, BondHarmonicType bondType) {
        float r = sqrtf(rSqr);
        float dr = r - bondType.r0;
        //printf("%f\n", (bondType.k/2.0) * 0.066 / (3.5*3.5));
        float eng = bondType.k * dr * dr * 0.5f;
        return 0.5f * eng; //0.5 for splitting between atoms
    }
};
#endif
