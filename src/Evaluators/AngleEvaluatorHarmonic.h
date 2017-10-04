#pragma once
#ifndef EVALUATOR_ANGLE_HARMONIC
#define EVALUATOR_ANGLE_HARMONIC

#include "cutils_math.h"
#include "Angle.h"
#define EPSILON 0.00001f
class AngleEvaluatorHarmonic {
public:

    //evaluator.force(theta, angleType, s, distSqrs, directors, invDotProd);
    inline __device__ float3 force(AngleHarmonicType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, int myIdxInAngle) {
        float dTheta = theta - angleType.theta0;
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.theta0, myIdxInAngle, type);
        

        float forceConst = angleType.k * dTheta;
        float a = -2.0f * forceConst * s;
        float a11 = a*c/distSqrs[0];
        float a12 = -a*invDistProd;
        float a22 = a*c/distSqrs[1];
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        if (myIdxInAngle==0) {
            return ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
        } else if (myIdxInAngle==1) {
            return ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -0.5; 
        } else {
            return ((directors[1] * a22) + (directors[0] * a12)) * 0.5;
        }


    }


    inline __device__ double3 force(AngleHarmonicType angleType, double theta, double s, double c, double distSqrs[2], double3 directors[2], double invDistProd, int myIdxInAngle) {
        double dTheta = theta - double(angleType.theta0);
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.theta0, myIdxInAngle, type);
        

        double forceConst = double(angleType.k) * dTheta;
        double a = -2.0f * forceConst * s;
        double a11 = a*c/distSqrs[0];
        double a12 = -a*invDistProd;
        double a22 = a*c/distSqrs[1];
        if (myIdxInAngle==0) {
            return ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
        } else if (myIdxInAngle==1) {
            return ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -0.5; 
        } else {
            return ((directors[1] * a22) + (directors[0] * a12)) * 0.5;
        }


    }





    inline __device__ void forcesAll(AngleHarmonicType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, float3 forces[3]) {
        float dTheta = theta - angleType.theta0;
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.theta0, myIdxInAngle, type);
        

        float forceConst = angleType.k * dTheta;
        float a = -2.0f * forceConst * s;
        float a11 = a*c/distSqrs[0];
        float a12 = -a*invDistProd;
        float a22 = a*c/distSqrs[1];
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        forces[0] = ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
        forces[1] = ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -0.5; 
        forces[2] = ((directors[1] * a22) + (directors[0] * a12)) * 0.5;


    }


    inline __device__ void forcesAll(AngleHarmonicType angleType, double theta, double s, double c, double distSqrs[2], double3 directors[2], double invDistProd, double3 forces[3]) {
        double dTheta = theta - double(angleType.theta0);
        //   printf("current %f theta eq %f idx %d, type %d\n", acosf(c), angleType.theta0, myIdxInAngle, type);
        

        double forceConst = double(angleType.k) * dTheta;
        double a = -2.0f * forceConst * s;
        double a11 = a*c/distSqrs[0];
        double a12 = -a*invDistProd;
        double a22 = a*c/distSqrs[1];
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        forces[0] = ((directors[0] * a11) + (directors[1] * a12)) * 0.5;
        forces[1] = ((directors[0] * a11) + (directors[1] * a12) + (directors[1] * a22) + (directors[0] * a12)) * -0.5; 
        forces[2] = ((directors[1] * a22) + (directors[0] * a12)) * 0.5;


    }




    inline __device__ float energy(AngleHarmonicType angleType, float theta, float3 directors[2]) {
        float dTheta = theta - angleType.theta0;
        return (1.0f / 6.0f) * dTheta * dTheta * angleType.k; // 1/6 comes from 1/3 (energy split between three atoms) and 1/2 from 1/2 k dtheta^2

    }
};

#endif

