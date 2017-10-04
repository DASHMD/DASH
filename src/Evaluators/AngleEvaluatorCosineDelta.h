#pragma once
#ifndef EVALUATOR_ANGLE_COSINE_DELTA_
#define EVALUATOR_ANGLE_COSINE_DELTA

#include "cutils_math.h"
#include "Angle.h"
#define EPSILON 0.00001f
class AngleEvaluatorCosineDelta{
public:

    //evaluator.force(theta, angleType, s, distSqrs, directors, invDotProd);
    inline __device__ float3 force(AngleCosineDeltaType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, int myIdxInAngle) {
        float cot = c / s;
        //float dTheta = theta - angleType.theta0;
        //float dCosTheta = cosf(dTheta);



        float a = -angleType.k;

        float a11 = a*c/distSqrs[0];
        float a12 = -a*invDistProd;
        float a22 = a*c/distSqrs[1];

        float b11 = -a*c*cot/distSqrs[0];
        float b12 = a*cot*invDistProd;
        float b22 = -a*c*cot/distSqrs[1];

        float c0 = cosf(angleType.theta0);
        float s0 = cosf(angleType.theta0);
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        if (myIdxInAngle==0) {
            return (directors[0] * a11 + directors[1] * a12) * c0 + (directors[0] * b11 + directors[1] * b12) * s0;
        } else if (myIdxInAngle==1) {
            return 
                (directors[0] * a11 + directors[1] * a12) * -c0 + (directors[0] * b11 + directors[1] * b12) * -s0 
                +
                (directors[1] * a22 + directors[0] * a12) * -c0 + (directors[1] * b22 + directors[0] * b12) * -s0
                ;

        } else {
            return (directors[1] * a22 + directors[0] * a12) * c0 + (directors[1] * b22 + directors[0] * b12) * s0;
        }


    }


    // double precision force routine
    inline __device__ double3 force(AngleCosineDeltaType angleType, double theta, double s, double c, double distSqrs[2], double3 directors[2], double invDistProd, int myIdxInAngle) {
        double cot = c / s;
        double a = -double(angleType.k);

        double a11 = a*c/distSqrs[0];
        double a12 = -a*invDistProd;
        double a22 = a*c/distSqrs[1];

        double b11 = -a*c*cot/distSqrs[0];
        double b12 = a*cot*invDistProd;
        double b22 = -a*c*cot/distSqrs[1];

        double c0 = cosf(angleType.theta0);
        double s0 = cosf(angleType.theta0);
        if (myIdxInAngle==0) {
            return (directors[0] * a11 + directors[1] * a12) * c0 + (directors[0] * b11 + directors[1] * b12) * s0;
        } else if (myIdxInAngle==1) {
            return 
                (directors[0] * a11 + directors[1] * a12) * -c0 + (directors[0] * b11 + directors[1] * b12) * -s0 
                +
                (directors[1] * a22 + directors[0] * a12) * -c0 + (directors[1] * b22 + directors[0] * b12) * -s0
                ;

        } else {
            return (directors[1] * a22 + directors[0] * a12) * c0 + (directors[1] * b22 + directors[0] * b12) * s0;
        }


    }



    inline __device__ void forcesAll(AngleCosineDeltaType angleType, float theta, float s, float c, float distSqrs[2], float3 directors[2], float invDistProd, float3 forces[3]) {
        float cot = c / s;
        //float dTheta = theta - angleType.theta0;
        //float dCosTheta = cosf(dTheta);



        float a = -angleType.k;

        float a11 = a*c/distSqrs[0];
        float a12 = -a*invDistProd;
        float a22 = a*c/distSqrs[1];

        float b11 = -a*c*cot/distSqrs[0];
        float b12 = a*cot*invDistProd;
        float b22 = -a*c*cot/distSqrs[1];

        float c0 = cosf(angleType.theta0);
        float s0 = cosf(angleType.theta0);
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        forces[0] = (directors[0] * a11 + directors[1] * a12) * c0 + (directors[0] * b11 + directors[1] * b12) * s0;
        forces[1] = (directors[0] * a11 + directors[1] * a12) * -c0 + (directors[0] * b11 + directors[1] * b12) * -s0 
            +
            (directors[1] * a22 + directors[0] * a12) * -c0 + (directors[1] * b22 + directors[0] * b12) * -s0
            ;

        forces[2] = (directors[1] * a22 + directors[0] * a12) * c0 + (directors[1] * b22 + directors[0] * b12) * s0;

        

    }

    inline __device__ void forcesAll(AngleCosineDeltaType angleType, double theta, double s, double c, double distSqrs[2], double3 directors[2], double invDistProd, double3 forces[3]) {
        double cot = c / s;
        //float dTheta = theta - angleType.theta0;
        //float dCosTheta = cosf(dTheta);



        double a = -double(angleType.k);

        double a11 = a*c/distSqrs[0];
        double a12 = -a*invDistProd;
        double a22 = a*c/distSqrs[1];

        double b11 = -a*c*cot/distSqrs[0];
        double b12 = a*cot*invDistProd;
        double b22 = -a*c*cot/distSqrs[1];

        double c0 = cos(double(angleType.theta0));
        double s0 = cos(double(angleType.theta0));
        //   printf("forceConst %f a %f s %f dists %f %f %f\n", forceConst, a, s, a11, a12, a22);
        //printf("hey %f, eq %f\n", theta, angleType.theta0);
        //printf("directors %f %f %f .. %f %f %f\n", directors[0].x, directors[0].y, directors[0].z,directors[1].x, directors[1].y, directors[1].z);
        //printf("a a11 a12 a22 %f %f %f %f\n", a, a11, a12, a22);
        forces[0] = (directors[0] * a11 + directors[1] * a12) * c0 + (directors[0] * b11 + directors[1] * b12) * s0;
        forces[1] = (directors[0] * a11 + directors[1] * a12) * -c0 + (directors[0] * b11 + directors[1] * b12) * -s0 
            +
            (directors[1] * a22 + directors[0] * a12) * -c0 + (directors[1] * b22 + directors[0] * b12) * -s0
            ;

        forces[2] = (directors[1] * a22 + directors[0] * a12) * c0 + (directors[1] * b22 + directors[0] * b12) * s0;



    }

    inline __device__ float energy(AngleCosineDeltaType angleType, float theta, float3 directors[2]) {
        float dTheta = theta - angleType.theta0;
        return (1.0f / 3.0f) * (1.0f - cosf(dTheta)); //energy split between three atoms
    }
};

#endif

