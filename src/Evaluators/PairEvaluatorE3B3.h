#pragma once

#ifndef EVALUATOR_E3B3
#define EVALUATOR_E3B3

#include "cutils_math.h"

// a class to evaluate the two-body correction term to the usual TIP4P/2005 model 
// for water;
//
// note that this is only calculated w.r.t. Oxygen-Oxygen interactions
//
// see:  J. Chem. Theory Comput. 2015, 11, 2268-2277
// for further details

class EvaluatorE3B3 {
    // our k2 constant (see reference) is 4.872 A^-1;
    // in real and LJ units, we use Angstroms; so, no unit conversion needed.
    float k2;

    // E2 constant is provided in kJ/mol in reference paper, which will require
    // some unit conversion depending on if state.units.setLJ() or .setReal() was called
    // ... so, get it from the constructor
    float E2;

    public:
        // asume that unit conversion was done prior to instantiation
        
        /* EvaluatorE3B3 Constructor
         * - takes E2 parameter, units already taken care of by this point
         * - distances when using the E3B3 model can be assumed to be in units of Angstroms,
         *   so no unit conversion is needed
         */
        EvaluatorE3B3(float E2_) {
            E2 = E2_;
            k2 = 4.872;
        }
        
        // takes input dr, the displacement vector $r_{ij}$
        inline __device__ float3 force(float3 dr) {
            float r = length(dr); // length from cutils_math.h
            float forceScalar = k2 * E2 * expf(-k2 * r) / r;
            return dr * forceScalar;
        }

        inline __device__ float energy(float3 dr) {
            float r = length(dr);
            // factor of 0.5 to account for double counting; otherwise, simple exponential expression
            return (0.5f * E2 * expf(-k2 * r));
        }
        
};
#endif /* EVALUATOR_E3B3 */

