#pragma once
#ifndef IMPROPER_HARMONIC_H
#include "Improper.h"
class ImproperEvaluatorHarmonic {
public:
    inline __device__ float dPotential(ImproperHarmonicType improperType, float theta) {
        float dTheta = theta - improperType.thetaEq;

        float dp = improperType.k * dTheta;
        return dp;
    }
    inline __device__ float potential(ImproperHarmonicType improperType, float theta) {
        float dTheta = theta - improperType.thetaEq;
        return (1.0f/2.0f) * dTheta * dTheta * improperType.k;

    }


};

#endif

