#pragma once
#ifndef DATACOMPUTERTEMPERATURE_H
#define DATACOMPUTERTEMPERATURE_H

#include "DataComputer.h"
#include "Virial.h"

namespace MD_ENGINE {
    class DataComputerTemperature : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t);

            void computeScalar_CPU();
            void computeVector_CPU();
            void computeTensor_CPU();
            void computeScalarFromTensor();
            void computeTensorFromScalar();

            DataComputerTemperature(State *, std::string computeMode_);
            int ndf;
            double totalKEScalar;
            double tempScalar;
            std::vector<double> tempVector;
            Virial tempTensor;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &);
            void appendTensor(boost::python::list &);

            double getScalar();
            Virial getTensor();

            void prepareForRun();

    };
};

#endif
