#pragma once
#ifndef DATACOMPUTERBOUNDS_H
#define DATACOMPUTERBOUNDS_H

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "Bounds.h"

namespace MD_ENGINE {
    class DataComputerBounds: public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t){};
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU();
            void computeVector_CPU(){};
            void computeTensor_CPU(){};

            DataComputerBounds(State *);
            void prepareForRun();
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &){};
            void appendTensor(boost::python::list &){};
            Bounds storedBounds;


    };
};

#endif
