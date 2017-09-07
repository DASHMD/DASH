#pragma once
#ifndef HELPERS_H
#define HELPERS_H

#include <array>
#include <vector>

#include <boost/variant.hpp>
#include <unordered_map>
#include <array>
#include "Virial.h"
#include "GPUArrayDeviceGlobal.h"
//#include "Atom.h"

template <class T, class K>
void cumulativeSum(T *data, K n) {
    int currentVal= 0;
    for (uint32_t i=0; i<n-1; i++) {
        int numInCell = data[i];
        data[i] = currentVal;
        currentVal += numInCell;
    }
    data[n-1] = currentVal; //okay, so now nth place has grid's starting Idx, n+1th place has ending
}
/*
            vals[0] = xx;
            vals[1] = yy;
            vals[2] = zz;
            vals[3] = xy;
            vals[4] = xz;
            vals[5] = yz;
            */
inline __device__ void computeVirial(Virial &v, float3 force, float3 dr) {
    v[0] += force.x * dr.x;
    v[1] += force.y * dr.y;
    v[2] += force.z * dr.z;
    v[3] += force.x * dr.y;
    v[4] += force.x * dr.z;
    v[5] += force.y * dr.z;
}


template <class SRCVar, class SRCBase, class SRCFull, class DEST, class TYPEHOLDER, int N>
int copyMultiAtomToGPU(int nAtoms, std::vector<SRCVar> &src, std::vector<int> &idToIdx, GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs, std::unordered_map<int, TYPEHOLDER> *forcerTypes, GPUArrayDeviceGlobal<TYPEHOLDER> *parameters, int maxExistingType) {
    std::vector<int> idxs(nAtoms+1, 0); //started out being used as counts
    std::vector<int> numAddedPerAtom(nAtoms, 0);
    bool redundantCalcs = N <= 3;
    std::vector<DEST> destHost;
    if (redundantCalcs ) {
        //so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
        for (SRCVar &sVar : src) {
            SRCFull &s = boost::get<SRCFull>(sVar);
            for (int i=0; i<N; i++) {
                int id = s.ids[i];
                idxs[idToIdx[id]]++;
            }

        }
        cumulativeSum(idxs.data(), nAtoms+1);
        destHost = std::vector<DEST> (idxs.back());
        for (SRCVar &sVar : src) {
            SRCFull &s = boost::get<SRCFull>(sVar);
            SRCBase *base = (SRCBase *) &s;
            std::array<int, N> atomIds = s.ids;
            std::array<int, N> atomIndexes;
            for (int i=0; i<N; i++) {
                atomIndexes[i] = idToIdx[atomIds[i]];
            }
            DEST d;
            d.takeIds(base);
            uint32_t type = s.type;
            assert(N <= 8); //three bits for which idx, allowing up to 8 member forcers, can be changed later
            for (int i=0; i<N; i++) {
                DEST dForIth = d;
                dForIth.type = type | (i << 29);
                destHost[idxs[atomIndexes[i]] + numAddedPerAtom[atomIndexes[i]]] = dForIth;
                numAddedPerAtom[atomIndexes[i]]++;
            }
        }
    } else {
        for (SRCVar &sVar : src) {
            SRCFull &s = boost::get<SRCFull>(sVar);
            SRCBase *base = (SRCBase *) &s;
            DEST d;
            d.takeIds(base);
            d.type = s.type;
            destHost.push_back(d);
        }

    }
    *dest = GPUArrayDeviceGlobal<DEST>(destHost.size());
    dest->set(destHost.data());
    int maxPerBlock = 0;
    if (redundantCalcs) {
        *destIdxs = GPUArrayDeviceGlobal<int>(idxs.size());
        destIdxs->set(idxs.data());

        //getting max # bonds per block
        for (int i=0; i<nAtoms; i+=PERBLOCK) {
            maxPerBlock = std::fmax(maxPerBlock, idxs[std::fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
        }
    }


    //now copy types
    //if user is silly and specifies huge types values, these kernels could crash
    //should add error messages and such about that
    std::vector<TYPEHOLDER> types(maxExistingType+1);
    for (auto it = forcerTypes->begin(); it!= forcerTypes->end(); it++) {
        types[it->first] = it->second;
    }
    *parameters = GPUArrayDeviceGlobal<TYPEHOLDER>(types.size());
    parameters->set(types.data());


    return maxPerBlock;
}

#endif
