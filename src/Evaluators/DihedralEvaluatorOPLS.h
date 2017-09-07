#pragma once
#ifndef EVALUATOR_DIHEDRAL_OPLS
#define EVALUATOR_DIHEDRAL_OPLS

#include "cutils_math.h"
#include "Dihedral.h"
class DihedralEvaluatorOPLS {
    public:
        //dihedralType, phi, c, scValues, invLenSqrs, c12Mags, c0,

                //float3 myForce = evaluator.force(dihedralType, phi, c, scValues, invLenSqrs, c12Mags, c0, c, invMagProds, c12Mags, invLens, directors, myIdxInDihedral);
        inline __device__ float dPotential(DihedralOPLSType dihedralType, float phi) {
    //LAMMPS pre-multiplies all of its coefs by 0.5.  We're doing it in the kernel.
            return -0.5 * (
                    dihedralType.coefs[0] * sinf(phi)
                    - 2.0f * dihedralType.coefs[1] * sinf(2.0f*phi) 
                    + 3.0f * dihedralType.coefs[2] * sinf(3.0f*phi)
                    - 4.0f * dihedralType.coefs[3] * sinf(4.0f*phi)
                    )
                ;
        }
        inline __device__ float potential(DihedralOPLSType dihedralType, float phi) {
            return  0.5 * (
                           dihedralType.coefs[0] * (1.0f + cosf(phi))
                           + dihedralType.coefs[1] * (1.0f - cosf(2.0f*phi))
                           + dihedralType.coefs[2] * (1.0f + cosf(3.0f*phi)) 
                           + dihedralType.coefs[3] * (1.0f - cosf(4.0f*phi)) 
                          );
        }
};

#endif


