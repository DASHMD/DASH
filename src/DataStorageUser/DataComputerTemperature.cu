#include "DataComputerTemperature.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
#include "Fix.h"
#include "Group.h"

namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerTemperature::DataComputerTemperature(State *state_, std::string computeMode_) : DataComputer(state_, computeMode_, false) {

}


void DataComputerTemperature::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
         accumulate_gpu<float, float4, SumVectorSqr3DOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBuffer.getDevData(), state->gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverW());
    } else {
        accumulate_gpu_if<float, float4, SumVectorSqr3DOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorSqr3DOverWIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
}


void DataComputerTemperature::prepareForRun() {
    DataComputer::prepareForRun();
}


void DataComputerTemperature::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();

    oneToOne_gpu<float, float4, SumVectorSqr3DOverW, 8> <<< NBLOCK(nAtoms / (double) 8), PERBLOCK>>> 
            (gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, SumVectorSqr3DOverW());
    if (transferToCPU) {
        gpuBuffer.dataToHost();
        gpd.ids.dataToHost();
    }

}

void DataComputerTemperature::computeTensor_GPU(bool transferToCPU, uint32_t groupTag) {
    GPUData &gpd = state->gpd;
    gpuBuffer.d_data.memset(0); 
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
        accumulate_gpu<Virial, float4, SumVectorToVirialOverW, N_DATA_PER_THREAD>  <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            ((Virial *) gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialOverW());    
    } else {
        accumulate_gpu_if<Virial, float4, SumVectorToVirialOverWIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(Virial)>>>
            ((Virial *) gpuBuffer.getDevData(), gpd.vs.getDevData(), nAtoms, state->devManager.prop.warpSize, SumVectorToVirialOverWIf(gpd.fs.getDevData(), groupTag));
    } 
    if (transferToCPU) {
        //does NOT sync
        gpuBuffer.dataToHost();
    }
}

void DataComputerTemperature::computeScalar_CPU() {
    
    //int n;
    double total = gpuBuffer.h_data[0];
    Group &thisGroup = state->groups[lastGroupTag];

    ndf = thisGroup.getNDF();
    
    totalKEScalar = total * state->units.mvv_to_eng; 
    tempScalar = state->units.mvv_to_eng * total / (state->units.boltz * ndf); 
}


void DataComputerTemperature::computeVector_CPU() {
    //appending members in group in no meaningful order
    std::vector<float> &kes = gpuBuffer.h_data;
    std::vector<uint> &ids = state->gpd.ids.h_data;
    std::vector<int> &idToIdxOnCopy = state->gpd.idToIdxsOnCopy;
    std::vector<Atom> &atoms = state->atoms;

    tempVector.erase(tempVector.begin(), tempVector.end());

    double conv = state->units.mvv_to_eng / state->units.boltz / 3.0;

    for (int i=0; i<kes.size(); i++) {
        int idx = idToIdxOnCopy[ids[i]];
        if (atoms[idx].groupTag & lastGroupTag) {
            tempVector.push_back(kes[i] * conv);
        }
    }
}

void DataComputerTemperature::computeTensor_CPU() {
    Virial total = *(Virial *) &gpuBuffer.h_data[0];
    total *= (state->units.mvv_to_eng / state->units.boltz);
    /*
       int n;
       if (lastGroupTag == 1) {
       n = state->atoms.size();
       } else {
       n = * (int *) &gpuBuffer.h_data[1];
       }
     */
    tempTensor = total;
}

void DataComputerTemperature::computeTensorFromScalar() {
    int zeroDim = 3;
    if (state->is2d) {
        zeroDim = 2;
        tempTensor[0] = tempTensor[1] = totalKEScalar / 2.0;
    } else {
        tempTensor[0] = tempTensor[1] = tempTensor[2] = totalKEScalar / 3.0;
    }
    for (int i=zeroDim; i<6; i++) {
        tempTensor[i] = 0;
    }

}

void DataComputerTemperature::computeScalarFromTensor() {
    //int n;
    /*
    if (lastGroupTag == 1) {
        n = state->atoms.size();//\* (int *) &gpuBuffer.h_data[1];
    } else {
        n = * (int *) &gpuBuffer.h_data[1];
    }
    */ //
    /*
    if (state->is2d) {
        ndf = 2*(n-1); //-1 is analagous to extra_dof in lammps
    } else {
        ndf = 3*(n-1);
    }
    */
    Group &thisGroup = state->groups[lastGroupTag];
    ndf = thisGroup.getNDF();
    //ndf = state->groups[lastGroupTag].getNDF();
    totalKEScalar = (tempTensor[0] + tempTensor[1] + tempTensor[2]) * state->units.boltz;
    tempScalar = totalKEScalar / ndf;


}

void DataComputerTemperature::appendScalar(boost::python::list &vals) {
    vals.append(tempScalar);
}
void DataComputerTemperature::appendVector(boost::python::list &vals) {
    vals.append(tempVector);
}
void DataComputerTemperature::appendTensor(boost::python::list &vals) {
    vals.append(tempTensor);
}


