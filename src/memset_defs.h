#pragma once
#ifndef MEMSET_DEFS_H
#define MEMSET_DEFS_H
#include "globalDefs.h"
__global__ void memsetByValList_32(int *, int, int);
__global__ void memsetByValList_64(int2 *, int2, int);
__global__ void memsetByValList_96(int3 *, int3, int);
__global__ void memsetByValList_128(int4 *, int4, int);


__global__ void memsetByValSurf_32(cudaSurfaceObject_t, int, int);
__global__ void memsetByValSurf_64(cudaSurfaceObject_t, int2, int);
__global__ void memsetByValSurf_96(cudaSurfaceObject_t, int3, int);
__global__ void memsetByValSurf_128(cudaSurfaceObject_t, int4, int);
#endif

