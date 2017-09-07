#pragma once
#ifndef FIX_HELPERS_H
#define FIX_HELPERS_H
#include "BoundsGPU.h"
//__device__ float3 harmonicForce(BoundsGPU bounds, float3 posSelf, float3 posOther, float k, float rEq);
inline __device__ float3 harmonicForce(BoundsGPU bounds, float3 posSelf, float3 posOther, float k, float rEq) {
    float3 bondVec = bounds.minImage(posSelf - posOther);
    float r = length(bondVec);
    float dr = r - rEq;
    float rk = k * dr;
    if (r > 0) {//MAKE SURE ALL THIS WORKS, I JUST BORROWED FROM LAMMPS
        float fBond = -rk/r;
        return bondVec * fBond;
    } 
    return make_float3(0, 0, 0);

}



inline __device__ float4 perAtomFromId(cudaTextureObject_t &idToIdxs, float4 *xs, int id) {
    int idx = tex2D<int>(idToIdxs, XIDX(id, sizeof(int)), YIDX(id, sizeof(int)));
    return xs[idx];
}

inline __device__ float4 float4FromIndex(cudaTextureObject_t &xs, int index) {
    return tex2D<float4>(xs, XIDX(index, sizeof(float4)), YIDX(index, sizeof(float4)));
}
#endif
