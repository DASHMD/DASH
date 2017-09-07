#pragma once
#ifndef EVALUATOR_DIHEDRAL_CHARMM
#define EVALUATOR_DIHEDRAL_CHARMM

#include "cutils_math.h"
#include "Dihedral.h"
class DihedralEvaluatorCHARMM {
    public:
        //dihedralType, phi, c, scValues, invLenSqrs, c12Mags, c0,

                //float3 myForce = evaluator.force(dihedralType, phi, c, scValues, invLenSqrs, c12Mags, c0, c, invMagProds, c12Mags, invLens, directors, myIdxInDihedral);
        inline __device__ float dPotential(DihedralCHARMMType dihedralType, float phi) {
            return dihedralType.k * dihedralType.n * sinf(dihedralType.d - dihedralType.n*phi);
        }
        inline __device__ float potential(DihedralCHARMMType dihedralType, float phi) {
            return dihedralType.k * (1 + cosf(dihedralType.n*phi - dihedralType.d));

        }

};

#endif


