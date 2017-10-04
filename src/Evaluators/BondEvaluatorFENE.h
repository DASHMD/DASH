#pragma once
#ifndef BONDEVALUATORFENE_H
#define BONDEVALUATORFENE_H

#include "Bond.h"

class BondEvaluatorFENE{
public:
    inline __device__ float3 force(float3 bondVec, float rSqr, BondFENEType bondType) {
        float k = bondType.k;
        float r0 = bondType.r0;
        float eps = bondType.eps;
        float sig = bondType.sig;
        float r0Sqr = r0*r0;
        float rlogarg = 1.0f - rSqr / r0Sqr;
        if (rlogarg < .1f) {
            if (rlogarg < -3.0f) {
                printf("FENE bond too long\n");
            }
            rlogarg = 0.1f;
        }
        float fbond = -k / rlogarg;
        if (rSqr < powf(2.0f, 1.0f/3.0f) * sig * sig) {
            float sr2 = sig*sig/rSqr;
            float sr6 = sr2*sr2*sr2;
            fbond += 48.0f*eps*sr6*(sr6-0.5f) / rSqr;

        }
        float3 force = bondVec * fbond;
        return force;
    }

    // double precision force routine
    inline __device__ double3 force(double3 bondVec, double rSqr, BondFENEType bondType) {
        double k = double(bondType.k);
        double r0 = double(bondType.r0);
        double eps = double(bondType.eps);
        double sig = double(bondType.sig);
        double r0Sqr = r0*r0;
        double rlogarg = 1.0f - rSqr / r0Sqr;
        if (rlogarg < .1f) {
            if (rlogarg < -3.0f) {
                printf("FENE bond too long\n");
            }
            rlogarg = 0.1f;
        }
        double fbond = -k / rlogarg;
        if (rSqr < pow(2.0f, 1.0f/3.0f) * sig * sig) {
            double sr2 = sig*sig/rSqr;
            double sr6 = sr2*sr2*sr2;
            fbond += 48.0f*eps*sr6*(sr6-0.5f) / rSqr;

        }
        double3 force = bondVec * fbond;
        return force;
    }





    inline __device__ float energy(float3 bondVec, float rSqr, BondFENEType bondType) {
        float k = bondType.k;
        float r0 = bondType.r0;
        float eps = bondType.eps;
        float sig = bondType.sig;
        float sigOverR2 = sig*sig/rSqr;
        float sigOverR6 = powf(sigOverR2, 3);
        float eng = -0.5f*k*r0*r0*logf(1.0f - rSqr / (r0 * r0)) + 4*eps*(sigOverR6*sigOverR6 - sigOverR6) + eps;
        return 0.5f * eng; //0.5 for splitting between atoms
    }
};
#endif
