#pragma once

#include "DataComputer.h"
#include "Virial.h"
class EvaluatorWrapper;
#include <string>


namespace MD_ENGINE {
    class DataComputerDipolarCoupling : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t){};
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU();
            void computeVector_CPU(){};
            void computeTensor_CPU(){};

            //computing coupling for A atoms coupling with atoms in group B
            //magnetogyric ratio should be in rad/(sec*tesla)
            DataComputerDipolarCoupling(State *, std::string computeMode_, std::string groupHandleA_, std::string groupHandleB_, double magnetoA_, double magnetoB_);
            uint32_t groupTagA;
            uint32_t groupTagB;
            std::string groupHandleA;
            std::string groupHandleB;
            GPUArrayGlobal<float> rCutSqrArray;
            GPUArrayGlobal<float> coalescedInvR6;
            std::vector<double> couplingsSqr;
            double magnetoA;
            double magnetoB;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &){};
            void appendTensor(boost::python::list &){};


            void prepareForRun();
            boost::shared_ptr<EvaluatorWrapper> evalWrap;

    };
};

