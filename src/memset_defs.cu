#include "memset_defs.h"
#define MEMSET_N(NAME, TYPE)\
__global__ void NAME (TYPE *dest, TYPE val, int n) {\
    int i = GETIDX();\
    if (i<n) {\
        dest[i] = val;\
    }\
}\


#define MEMSETSURF_N(NAME, TYPE)\
__global__ void NAME (cudaSurfaceObject_t surf, TYPE val, int n) {\
    int i = GETIDX();\
    if (i<n) {\
        surf2Dwrite(val, surf, sizeof(TYPE) * XIDX(i, sizeof(TYPE)), YIDX(i, sizeof(TYPE)));\
    }\
}


MEMSET_N(memsetByValList_32, int);
MEMSET_N(memsetByValList_64, int2);
MEMSET_N(memsetByValList_96, int3);
MEMSET_N(memsetByValList_128, int4);

MEMSETSURF_N(memsetByValSurf_32, int);
MEMSETSURF_N(memsetByValSurf_64, int2);
MEMSETSURF_N(memsetByValSurf_128, int4); //NO int3 for surfs
