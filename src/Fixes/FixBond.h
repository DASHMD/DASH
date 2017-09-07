#pragma once 
#ifndef FIXBOND_H
#define FIXBOND_H

#include <array>
#include <unordered_map>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python/list.hpp>

#include "Fix.h"
#include "Bond.h"
#include "State.h"
#include "helpers.h"  // cumulative sum
#include "TypedItemHolder.h"
#include <unordered_map>
#include "VariantPyListInterface.h"



template <class SRC, class DEST, class BONDTYPEHOLDER>
int copyBondsToGPU(std::vector<Atom> &atoms, 
                   std::vector<BondVariant> &src, std::vector<int> &idToIdx,
                   GPUArrayDeviceGlobal<DEST> *dest, GPUArrayDeviceGlobal<int> *destIdxs, 
                   GPUArrayDeviceGlobal<BONDTYPEHOLDER> *parameters, int maxExistingType, std::unordered_map<int, BONDTYPEHOLDER> &bondTypes) {

    std::vector<int> idxs(atoms.size()+1, 0);  // started out being used as counts
    std::vector<int> numAddedPerAtom(atoms.size(), 0);

    // so I can arbitrarily order.  I choose to do it by the the way atoms happen to be sorted currently.  Could be improved.
    for (BondVariant &sVar : src) {
        SRC &s = boost::get<SRC>(sVar);
        for (int i=0; i<2; i++) {
            idxs[idToIdx[s.ids[i]]] ++;
        }
    }
    cumulativeSum(idxs.data(), atoms.size()+1);  
    std::vector<DEST> destHost(idxs.back());
    for (BondVariant &sv : src) {
        SRC &s = boost::get<SRC>(sv);
        std::array<int, 2> atomIds = s.ids;
        std::array<int, 2> atomIndexes;
        for (int i=0; i<2; i++) {
            atomIndexes[i] = idToIdx[atomIds[i]];
        }
        for (int i=0; i<2; i++) {
            DEST a;
            a.myId = atomIds[i];
            a.otherId = atomIds[!i];
            a.type = s.type;
            destHost[idxs[atomIndexes[i]] + numAddedPerAtom[atomIndexes[i]]] = a;
            numAddedPerAtom[atomIndexes[i]]++;
        }
    }
    *dest = GPUArrayDeviceGlobal<DEST>(destHost.size());
    dest->set(destHost.data());
    *destIdxs = GPUArrayDeviceGlobal<int>(idxs.size());
    destIdxs->set(idxs.data());

    //getting max # bonds per block
    int maxPerBlock = 0;
    for (uint32_t i=0; i<atoms.size(); i+=PERBLOCK) {
        maxPerBlock = std::fmax(maxPerBlock, idxs[std::fmin(i+PERBLOCK+1, idxs.size()-1)] - idxs[i]);
    }



    //now copy types
    //if user is silly and specifies huge types values, these kernels could crash
    //should add error messages and such about that
    std::vector<BONDTYPEHOLDER> types(maxExistingType+1);
    for (auto it = bondTypes.begin(); it!= bondTypes.end(); it++) {
        types[it->first] = it->second;
    }
    *parameters = GPUArrayDeviceGlobal<BONDTYPEHOLDER>(types.size());
    parameters->set(types.data());
    return maxPerBlock;

}

template <class CPUMember, class GPUMember, class BONDTYPEHOLDER>
class FixBond : public Fix, public TypedItemHolder {
    public:
        GPUArrayDeviceGlobal<GPUMember> bondsGPU;
        GPUArrayDeviceGlobal<int> bondIdxs;
        GPUArrayDeviceGlobal<BONDTYPEHOLDER> parameters; 
        std::vector<BondVariant> bonds;
        boost::python::list pyBonds;
        VariantPyListInterface<BondVariant, CPUMember> pyListInterface;
        int sharedMemSizeForParams;
        bool usingSharedMemForParams;

        int maxBondsPerBlock;
        std::unordered_map<int, BONDTYPEHOLDER> bondTypes;
        
        FixBond(SHARED(State) state_, std::string handle_, std::string groupHandle_, std::string type_,
                bool forceSingle_, int applyEvery_)
            : Fix(state_, handle_, groupHandle_, type_, forceSingle_, false, false, applyEvery_), pyListInterface(&bonds, &pyBonds) {
            maxBondsPerBlock = 0;
        }

        void setBondType(int n, CPUMember &forcer) {
            if (n<0) {
                std::cout << "Tried to set bonded potential for invalid type " << n << std::endl;
                assert(n>=0);
            }
            bondTypes[n] = forcer;
        }


        virtual bool prepareForRun() {
            std::vector<Atom> &atoms = state->atoms;

            int maxExistingType = -1;
            std::unordered_map<BONDTYPEHOLDER, int> reverseMap;
            for (auto it=bondTypes.begin(); it!=bondTypes.end(); it++) {
                maxExistingType = std::fmax(it->first, maxExistingType);
                reverseMap[it->second] = it->first;
            }

            for (BondVariant &bondVar : bonds) { //collecting un-typed bonds into types
                CPUMember &bond = boost::get<CPUMember>(bondVar);
                if (bond.type == -1) {
                    //cout << "gotta do" << endl;
                    //cout << "max existing type " << maxExistingType  << endl;
                    BONDTYPEHOLDER typeHolder = *(BONDTYPEHOLDER *) (&bond);
                    bool parameterFound = reverseMap.find(typeHolder) != reverseMap.end();
                    //std::cout << "is found " << parameterFound << std::endl;
                    if (parameterFound) {
                        bond.type = reverseMap[typeHolder];
                    } else {
                        maxExistingType+=1;
                        bondTypes[maxExistingType] = typeHolder;
                        reverseMap[typeHolder] = maxExistingType;
                        bond.type = maxExistingType;
                    //    std::cout << "assigning type of " << bond.type << std::endl;

                    }
                } 
            }
            maxBondsPerBlock = copyBondsToGPU<CPUMember, GPUMember, BONDTYPEHOLDER>(
                    atoms, bonds, state->idToIdx, &bondsGPU, &bondIdxs, &parameters, maxExistingType, bondTypes);
           // maxbondsPerBlock = copyMultiAtomToGPU<CPUVariant, CPUBase, CPUMember, GPUMember, ForcerTypeHolder, N>(state->atoms.size(), forcers, state->idToIdx, &forcersGPU, &forcerIdxs, &forcerTypes, &parameters, maxExistingType);
            setSharedMemForParams();
            return true;
        } 



        
        
        std::vector<int> getTypeIds() {
            std::vector<int> ids;
            for (auto it=bondTypes.begin(); it!=bondTypes.end(); it++) {
                ids.push_back(it->first);
            }
            return ids;
        }
        void duplicateMolecule(std::vector<int> &oldIds, std::vector<std::vector<int> > &newIds) {
            int ii = bonds.size();
            std::vector<CPUMember> belongingToOld;
            for (int i=0; i<ii; i++) {
                CPUMember &b = boost::get<CPUMember>(bonds[i]);
                std::array<int, 2> &ids = b.ids;
                bool found = false;
                for (int j=0; j<2; j++) {
                    if (find(oldIds.begin(), oldIds.end(), ids[j]) != oldIds.end()) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    belongingToOld.push_back(b);
                }
            }
            for (uint i=0; i<newIds.size(); i++) {
                for (uint j=0; j<belongingToOld.size(); j++) {
                    CPUMember copy = belongingToOld[j];
                    std::array<int, 2> idsNew = copy.ids;
                    for (int k=0; k<2; k++) {
                        auto it = find(oldIds.begin(), oldIds.end(), idsNew[k]);
                        if (it != oldIds.end()) {
                            idsNew[k] = newIds[i][it - oldIds.begin()];
                        }
                    }
                    copy.ids = idsNew;
                    bonds.push_back(copy);
                    pyListInterface.updateAppendedMember(false);
                }
            }
            pyListInterface.requestRefreshPyList();
        }
        /*
        void duplicateMolecule(std::map<int, int> &oldToNew) {
            int ii = bonds.size();
            for (int i=0; i<ii; i++) {
                CPUMember &b = boost::get<CPUMember>(bonds[i]);
                std::array<int, 2> &ids = b.ids;
                std::array<int, 2> idsNew = ids;
                for (int j=0; j<2; j++) {
                    if (oldToNew.find(ids[j]) != oldToNew.end()) {
                        idsNew[j] = oldToNew[ids[j]];
                    }
                }
                if (ids != idsNew) {
                    CPUMember duplicate = b;
                    duplicate.ids = idsNew;
                    bonds.push_back(duplicate);
                    pyListInterface.updateAppendedMember(false);


                }
            }
            pyListInterface.requestRefreshPyList();

            
        }
        */


        void setSharedMemForParams() {
            int size = parameters.size() * sizeof(BONDTYPEHOLDER);
            if (size + maxBondsPerBlock * sizeof(GPUMember) > state->devManager.prop.sharedMemPerBlock) {
                usingSharedMemForParams = false;
                sharedMemSizeForParams = 0;
            } else {
                usingSharedMemForParams = true;
                sharedMemSizeForParams = size;
            }

        }
        void deleteAtom(Atom *a) {
            int deleteId = a->id;
            for (int i=bonds.size()-1; i>=0; i--) {
                CPUMember &forcer= boost::get<CPUMember>(bonds[i]);
                bool deleteForcer = false;
                for (int id : forcer.ids) {
                    if (id == deleteId) {
                        deleteForcer = true;
                        break;
                    }
                }
                if (deleteForcer) {
                    bonds.erase(bonds.begin()+i, bonds.begin()+i+1);
                    pyListInterface.removeMember(i);
                    pyListInterface.requestRefreshPyList();
                }
            }
        }

        void updateForPIMD(int nPerRingPoly) {
            std::vector<BondVariant> RPbonds(bonds.size()*nPerRingPoly);
            for (int i=0; i<bonds.size(); i++) {
                CPUMember  asType = boost::get<CPUMember>(bonds[i]);
                for (int j=0; j<nPerRingPoly; j++) {
                    CPUMember RPcopy = asType;                      // create copy of the forcer member
                    RPcopy.ids[0] = asType.ids[0]*nPerRingPoly + j; // new id for RP atom1 in bond
                    RPcopy.ids[1] = asType.ids[1]*nPerRingPoly + j; // new id for RP atom2 in bond
                    RPbonds[i*nPerRingPoly+j] = RPcopy;             // place new member for RP bonds
                    if (j > 0 ) {pyListInterface.updateAppendedMember(false);}
                }
            }
            bonds = RPbonds;    // update the forcers
            pyListInterface.requestRefreshPyList(true);
        }

};

#endif
