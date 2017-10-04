#pragma once
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include "Python.h"
#include <vector>
#include <map>
#include "globalDefs.h"
#include <boost/shared_ptr.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <iostream>
#include "boost_for_export.h"
#include "BoundsGPU.h"
#include "Virial.h"
#include "GPUArrayGlobal.h"

class State;
namespace MD_ENGINE {

    class DataComputer {
    public:
        State *state;
        uint32_t lastGroupTag;
        //note that transfer flag doesn't make it sync.  

        //used for computers which are shared between multiple fixes.  Only need to have one computer per sim
        virtual void computeScalar_GPU(bool transferToCPU, uint32_t groupTag) = 0;
        virtual void computeVector_GPU(bool transferToCPU, uint32_t groupTag) = 0;
        virtual void computeTensor_GPU(bool transferToCPU, uint32_t groupTag) = 0;

        //after values transferred, can compute final value on host and store interally.  Retrieval implemented by each compute
        virtual void computeScalar_CPU() = 0;
        virtual void computeVector_CPU() = 0;
        virtual void computeTensor_CPU() = 0;






        virtual void appendScalar(boost::python::list &) = 0;
        virtual void appendVector(boost::python::list &) = 0;
        virtual void appendTensor(boost::python::list &) = 0;

        bool requiresVirials;
        bool requiresPerAtomVirials;

        GPUArrayGlobal<float> gpuBuffer; //will be cast as virial if necessary
        GPUArrayGlobal<float> gpuBufferReduce; //target for reductions, also maybe cast as virial
        std::vector<double> sorted; //sorted vector values

        std::string computeMode;
        virtual void prepareForRun();
        int dataMultiple;
        void compute_GPU(bool transferToCPU, uint32_t groupTag);
        void compute_CPU();
        void appendData(boost::python::list &);
        DataComputer(){};
        DataComputer(State *, std::string computeMode_, bool requiresVirials_);


    };

}
template<class SRC, class DST>
void sortToCPUOrder(std::vector<SRC> &src, std::vector<DST> &sorted, std::vector<uint> &ids, std::vector<int> &idToIdxs) {
    for (int i=0; i<src.size(); i++) {
        sorted[idToIdxs[ids[i]]] = src[i];
    }
}
