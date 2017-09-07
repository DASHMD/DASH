#include "DataComputerEnergy.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerEnergy::DataComputerEnergy(State *state_, py::list fixes_, std::string computeMode_, std::string groupHandleB_) : DataComputer(state_, computeMode_, false), groupHandleB(groupHandleB_) {

    groupTagB = state->groupTagFromHandle(groupHandleB);
    otherIsAll = groupHandleB == "all";
    if (py::len(fixes_)) {
        int len = py::len(fixes_);
        for (int i=0; i<len; i++) {
            py::extract<boost::shared_ptr<Fix> > fixPy(fixes_[i]);
            if (!fixPy.check()) {
                assert(fixPy.check());
            }
            fixes.push_back(fixPy);
        }
    }

}


void DataComputerEnergy::computeScalar_GPU(bool transferToCPU, uint32_t groupTag) {
    gpuBuffer.d_data.memset(0);
    gpuBufferReduce.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperMode("self");
        fix->setEvalWrapper();
        if (otherIsAll) {
            fix->singlePointEng(gpuBuffer.getDevData());
        } else {
            fix->singlePointEngGroupGroup(gpuBuffer.getDevData(), groupTag, groupTagB);

        }
        fix->setEvalWrapperMode("offload");
        fix->setEvalWrapper();
    }
    if (groupTag == 1 or !otherIsAll) { //if other isn't all, then only group-group energies got computed so need to sum them all up anyway.  If other is all then every eng gets computed so need to accumulate only things in group
         accumulate_gpu<float, float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingle());
    } else {
        accumulate_gpu_if<float, float, SumSingleIf, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float)>>>
            (gpuBufferReduce.getDevData(), gpuBuffer.getDevData(), nAtoms, state->devManager.prop.warpSize, SumSingleIf(gpd.fs.getDevData(), groupTag));
    }
    if (transferToCPU) {
        //does NOT sync
        gpuBufferReduce.dataToHost();
    }
}


void DataComputerEnergy::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    gpuBuffer.d_data.memset(0);
    lastGroupTag = groupTag;
    int nAtoms = state->atoms.size();

    for (boost::shared_ptr<Fix> fix : fixes) {
        fix->setEvalWrapperMode("self");
        fix->setEvalWrapper();
        if (otherIsAll) {
            fix->singlePointEng(gpuBuffer.getDevData());
        } else {
            fix->singlePointEngGroupGroup(gpuBuffer.getDevData(), groupTag, groupTagB);
        }
        fix->setEvalWrapperMode("offload");
        fix->setEvalWrapper();
    }
    if (transferToCPU) {
        gpuBuffer.dataToHost();
    }
}




void DataComputerEnergy::computeScalar_CPU() {
    //int n;
    double total = gpuBufferReduce.h_data[0];
    /*
    if (lastGroupTag == 1) {
        n = state->atoms.size();//* (int *) &tempGPUScalar.h_data[1];
    } else {
        n = * (int *) &gpuBufferReduce.h_data[1];
    }
    */
    //just going with total energy value, not average
    engScalar = total;
}

void DataComputerEnergy::computeVector_CPU() {
    //ids have already been transferred, look in doDataComputation in integUtil
    std::vector<uint> &ids = state->gpd.ids.h_data;
    std::vector<float> &src = gpuBuffer.h_data;
    sortToCPUOrder(src, sorted, ids, state->gpd.idToIdxsOnCopy);
}



void DataComputerEnergy::appendScalar(boost::python::list &vals) {
    vals.append(engScalar);
}
void DataComputerEnergy::appendVector(boost::python::list &vals) {
    vals.append(sorted);
}

void DataComputerEnergy::prepareForRun() {
    if (fixes.size() == 0) {
        fixes = state->fixesShr; //if none specified, use them all
    } else {
        //make sure that fixes are activated
        for (auto fix : fixes) {
            bool found = false;
            for (auto fix2 : state->fixesShr) {
                if (fix->handle == fix2->handle && fix->type == fix2->type) {
                    found = true;
                }
            }
            if (not found) {
                std::cout << "Trying to record energy for inactive fix " << fix->handle << ".  Quitting" << std::endl;
                assert(found);
            }
        }
    }
    DataComputer::prepareForRun();
}

