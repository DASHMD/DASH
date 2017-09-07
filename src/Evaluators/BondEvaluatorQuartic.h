#pragma once
#ifndef BONDEVALUATORQUARTIC_H
#define BONDEVALUATORQUARTIC_H

#include "Bond.h"

class BondEvaluatorQuartic {
public:
    inline __device__ float3 force(float3 bondVec, float rSqr, BondQuarticType bondType) {
        float r = sqrtf(rSqr);
        if (r > 0) {
            float dr = r - bondType.r0;
            float dr2= dr*dr;
            float dr3= dr2*dr;
            float dUdr = 2.0f*bondType.k2*dr + 3.0f*bondType.k3*dr2 + 4.0f*bondType.k4*dr3;
            float fBond = -dUdr/r;
            return bondVec * fBond;
        } 
        return make_float3(0, 0, 0);
    }
    inline __device__ float energy(float3 bondVec, float rSqr, BondQuarticType bondType) {
        float r  = sqrtf(rSqr);
        float dr = r - bondType.r0;
        float dr2= dr*dr;
        float dr3= dr2*dr;
        float dr4= dr2*dr2;
        float eng = bondType.k2*dr2 + bondType.k3*dr3 + bondType.k4*dr4;
        return 0.5f * eng; //0.5 for splitting between atoms
    }
};
#endif
