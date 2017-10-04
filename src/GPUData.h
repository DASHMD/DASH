#pragma once
#ifndef GPUDATA_H
#define GPUDATA_H

#include <map>

#include "GPUArrayGlobal.h"
#include "GPUArrayPair.h"
#include "GPUArrayDeviceGlobal.h"
#include "Virial.h"

class GPUData {

public:
    /* types (ints) are bit cast into the w value of xs.  Cast as int pls */
    GPUArrayPair<float4> xs;
    /* mass is stored in w value of vs.  ALWAYS do arithmetic as float3s, or
     * you will mess up id or mass */
    GPUArrayPair<float4> vs;
    /* groupTags (uints) are bit cast into the w value of fs */
    GPUArrayPair<float4> fs;
    GPUArrayPair<uint> ids;
    GPUArrayPair<float> qs;
    GPUArrayGlobal<int> idToIdxs;
    GPUArrayGlobal<Virial> virials;

    GPUArrayGlobal<float4> xsBuffer;
    GPUArrayGlobal<float4> vsBuffer;
    GPUArrayGlobal<float4> fsBuffer;
    GPUArrayGlobal<uint> idsBuffer;

    /* for transfer between GPUs */
    std::map<int, GPUArrayGlobal<float4>> neighborValBuffers;
    std::map<int, GPUArrayGlobal<uint>> neighborIdsBuffers;
    std::map<int, GPUArrayGlobal<int>> neighborIdxBuffers;

    std::map<int, GPUArrayGlobal<float4>> movedValBuffers;
    std::map<int, GPUArrayGlobal<uint>> movedIdsBuffers;
    std::map<int, GPUArrayGlobal<int>> movedIdxBuffers;

    /* duplicates of PIMD variables used by GridGPU */
    // nPerRingPoly; immediately after instantiation of GPUData, set this to the value for this gpd instance
    // int nPerRingPoly = 1;

    std::vector<int> idToIdxsOnCopy;

    // OMG REMEMBER TO ADD EACH NEW ARRAY TO THE ACTIVE DATA LIST IN INTEGRATOR OR PAIN AWAITS

    GPUData() {   }

    unsigned int activeIdx() {
        return xs.activeIdx;
    }
    unsigned int switchIdx() {
        /*! \todo Find a better way to keep track of all data objects */
        xs.switchIdx();
        vs.switchIdx();
        fs.switchIdx();
        ids.switchIdx();
        return qs.switchIdx();
    }

    unsigned int switchIdx(bool onlyPositions) {
        xs.switchIdx();
        return ids.switchIdx();
    }   

};

#endif
