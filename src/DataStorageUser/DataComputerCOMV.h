#pragma once
#ifndef DATACOMPUTERCOMV_H
#define DATACOMPUTERCOMV_H

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"

// a data computer for the center of mass velocity
namespace MD_ENGINE {
    class DataComputerCOMV : public DataComputer {
        public:


            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t){};
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU();
            void computeVector_CPU(){};
            void computeTensor_CPU(){};

            DataComputerCOMV(State *);

            void prepareForRun();
            
            // as from FixLinearMomentum
            GPUArrayGlobal<float4> sumMomentum;
            float4 systemMomentum;

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &){};
            void appendTensor(boost::python::list &){};


    };
};

#endif
