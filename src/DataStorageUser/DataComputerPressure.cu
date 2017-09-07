#include "DataComputerPressure.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerPressure::DataComputerPressure(State *state_, std::string computeMode_) : DataComputer(state_, computeMode_, true), tempComputer(state_, computeMode_) {
    usingExternalTemperature = false;
    if (computeMode == "vector") {
        requiresPerAtomVirials = true;
    }
}


void DataComputerPressure::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    mdAssert(groupTag == 1, "Trying to compute pressure for group other than 'all'");
    if (!usingExternalTemperature) {
        tempComputer.computeScalar_GPU(transferToCPU, groupTag);
    }
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
         accumulate_gpu<float, Virial, SumVirialToScalar, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBuffer.getDevData(), gpd.virials.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVirialToScalar());
    } else {
        accumulate_gpu_if<float, Virial, SumVirialToScalarIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBuffer.getDevData(), 
             gpd.virials.getDevData(), 
             nAtoms, 
             state->devManager.prop.warpSize, 
             SumVirialToScalarIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
}

void DataComputerPressure::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    if (transferToCPU) {
        state->gpd.virials.dataToHost();
    }
}



void DataComputerPressure::computeTensor_GPU(bool transferToCPU, uint32_t groupTag) {
    mdAssert(groupTag == 1, "Trying to compute pressure for group other than 'all'");
    if (!usingExternalTemperature) {
        tempComputer.computeTensor_GPU(transferToCPU, groupTag);
    }
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0); 
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
        
        accumulate_gpu<Virial, Virial, SumVirial, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            ((Virial *) gpuBuffer.getDevData(), gpd.virials.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVirial());    
    } else {
        accumulate_gpu_if<Virial, Virial, SumVirialIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            ((Virial *) gpuBuffer.getDevData(), gpd.virials.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVirialIf(gpd.fs.getDevData(), groupTag));
    } 
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
}

void DataComputerPressure::computeScalar_CPU() {
    //we are assuming that z component of virial is zero if sim is 2D
    float boltz = state->units.boltz;
    double tempScalar_loc, ndf_loc;
    if (usingExternalTemperature) {
        tempScalar_loc = tempScalar;
        ndf_loc = tempNDF;
    } else {
        tempComputer.computeScalar_CPU();
        tempScalar_loc = tempComputer.tempScalar;
        ndf_loc = tempComputer.ndf;
    }
    double sumVirial = gpuBuffer.h_data[0];
    double dim = state->is2d ? 2 : 3;
    double volume = state->boundsGPU.volume();
    pressureScalar = (tempScalar_loc * ndf_loc * boltz + sumVirial) / (dim * volume) * state->units.nktv_to_press;
    //printf("heyo, scalar %f conv %f\n", pressureScalar, state->units.nktv_to_press);
}


void DataComputerPressure::computeVector_CPU() {
   //not implemented 
    assert(false);
}

void DataComputerPressure::computeTensor_CPU() {
    Virial tempTensor_loc;
    if (usingExternalTemperature) {
        tempTensor_loc = tempTensor;
    } else {
        tempComputer.computeTensor_CPU();
        tempTensor_loc = tempComputer.tempTensor;
    }
    pressureTensor = Virial(0, 0, 0, 0, 0, 0);
    Virial sumVirial = * (Virial *) gpuBuffer.h_data.data();
    double volume = state->boundsGPU.volume();
    for (int i=0; i<6; i++) {
        pressureTensor[i] = (tempTensor_loc[i] + sumVirial[i]) / volume * state->units.nktv_to_press;
    }
    if (state->is2d) {
        pressureTensor[2] = 0;
        pressureTensor[4] = 0;
        pressureTensor[5] = 0;
    }
}

void DataComputerPressure::appendScalar(boost::python::list &vals) {
    vals.append(pressureScalar);
}

void DataComputerPressure::appendVector(boost::python::list &vals) {
    //not implemented
    assert(false);
}

void DataComputerPressure::appendTensor(boost::python::list &vals) {
    vals.append(pressureTensor);
}

void DataComputerPressure::prepareForRun() {
    tempComputer = DataComputerTemperature(state, computeMode);
    tempComputer.prepareForRun();
    DataComputer::prepareForRun();
}

