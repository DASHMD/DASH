#pragma once

#include "DataComputer.h"
#include "GPUArrayGlobal.h"
#include "Fix.h"

class EvaluatorWrapper;

namespace MD_ENGINE {
    class DataComputerEField: public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t){};
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU(){};
            void computeVector_CPU();
            void computeTensor_CPU(){};

            double cutoff;
            DataComputerEField(State *, double cutoff_);
            void prepareForRun();


            void appendScalar(boost::python::list &){};
            void appendVector(boost::python::list &); 
            void appendTensor(boost::python::list &){};
            boost::shared_ptr<EvaluatorWrapper> evalWrap;


    };
};

