#pragma once
#ifndef EVALUATOR_ANGLE_CHARMM
#define EVALUATOR_ANGLE_CHARMM

#include "cutils_math.h"
#include "Angle.h"
#define EPSILON 0.00001f
class AngleEvaluatorCHARMM {
public:

    //evaluator.force(theta, angleType, s, distSqrs, directors, invDotProd);
    inline __device__ float3 force(AngleCHARMMType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, int myIdxInAngle) {
        float dTheta = theta - angleType.theta0;
        float forceConst = angleType.k * dTheta;
        float a     = -forceConst * s;
        float a11   = a*c/distSqrs[0];
        float a12   = -a*invDistProd;
        float a22   = a*c/distSqrs[1];

        // Added code for computing Urey-Bradley component (MW)
        float3 dr31;
        dr31        = directors[1] - directors[0]; // Urey-Bradley bond between 1 and 3 atoms
        float rsq31 = dot(dr31,dr31);
        float r31   = sqrtf(rsq31);
        float drub  = r31 - angleType.rub;
        float rk    = angleType.kub * drub;
        float fub   = -rk / r31;   // consider safe-checking for r31 > 0.0 ?
        //printf("Eang= %f\n", (1.0f / 2.0f) *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ));
        // End added code
        
        if (myIdxInAngle==0) {
            return (directors[0] * a11) + (directors[1] * a12) - dr31 * fub ;
        } else if (myIdxInAngle==1) {
            return ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12))*-1.0f ; 
        } else {
            return (directors[1] * a22) + (directors[0] * a12) + dr31 * fub ;
        }


    }
    inline __device__ void forcesAll(AngleCHARMMType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, float3 forces[3]) {
        float dTheta = theta - angleType.theta0;
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.theta0, myIdxInAngle, type);
        

        float forceConst = angleType.k * dTheta;
        float a = - forceConst * s;
        float a11 = a*c/distSqrs[0];
        float a12 = -a*invDistProd;
        float a22 = a*c/distSqrs[1];
        // Added code for computing Urey-Bradley component (MW)
        float3 dr31;
        dr31        = directors[1] - directors[0]; // Urey-Bradley bond between 1 and 3 atoms
        float rsq31 = dot(dr31,dr31);
        float r31   = sqrtf(rsq31);
        float drub  = r31 - angleType.rub;
        float rk    = angleType.kub * drub;
        float fub   = -rk / r31;   // consider safe-checking for r31 > 0.0 ?
        // End added code
        forces[0] = (directors[0] * a11) + (directors[1] * a12) - dr31 * fub ;
        forces[1] = ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -1.0f ; 
        forces[2] = (directors[1] * a22) + (directors[0] * a12) + dr31 * fub ;


    }

    inline __device__ float energy(AngleCHARMMType angleType, float theta, float3 directors[2]) {
        float dTheta = theta - angleType.theta0;
        float3 dr31;
        dr31        = directors[1] - directors[0]; // Urey-Bradley bond between 1 and 3 atoms
        float rsq31 = dot(dr31,dr31);
        float r31   = sqrtf(rsq31);
        float drub  = r31 - angleType.rub;
//        printf("theta = %f, r31 = %f, theta = %f,k = %f,rub = %f, kub = %f\n",theta,r31,angleType.theta0,angleType.k,angleType.rub,angleType.kub);
//        printf("eang = %f\n", 0.5f *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ) );
        return (1.0f / 6.0f) *( dTheta * dTheta * angleType.k + drub * drub * angleType.kub ) ; // 1/6 comes from 1/3 (energy split between three atoms) and 1/2 from 1/2 k dtheta^2

    }
};

#endif

