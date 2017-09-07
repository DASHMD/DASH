#pragma once
#ifndef DATACOMPUTERPRESSURE_H
#define DATACOMPUTERPRESSURE_H

#include "DataComputer.h"
#include "DataComputerTemperature.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"

namespace MD_ENGINE {
    class DataComputerPressure : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t);
            void computeTensor_GPU(bool, uint32_t);

            void computeScalar_CPU();
            void computeVector_CPU();
            void computeTensor_CPU();

            DataComputerPressure(State *, std::string);
            void prepareForRun();
            double pressureScalar;
            bool usingExternalTemperature;
            double tempScalar; //if using externaltemp, then these must be set each time you go to compute pressure
            Virial tempTensor; //if using externaltemp, then these must be set each time you go to compute pressure
            double tempNDF;
            Virial pressureTensor;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &);
            void appendTensor(boost::python::list &);

            double getScalar();
            Virial getTensor();
            DataComputerTemperature tempComputer;

    };
};

#endif
