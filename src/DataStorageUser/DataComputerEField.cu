#include "DataComputerEField.h"
#include "cutils_func.h"
#include "boost_for_export.h"
#include "State.h"

#include "EvaluatorWrapper.h"
#include "PairEvaluatorEField.h"
#include "PairEvaluatorNone.h"
namespace py = boost::python;
using namespace MD_ENGINE;

DataComputerEField::DataComputerEField(State *state_, double cutoff_) : DataComputer(state_, "vector", false), cutoff(cutoff_) {
    dataMultiple = 4;
}

void DataComputerEField::prepareForRun() {
    evalWrap = boost::shared_ptr<EvaluatorWrapper> (dynamic_cast<EvaluatorWrapper *>( new EvaluatorWrapperImplement<EvaluatorNone, false, 1, EvaluatorEField, true>(EvaluatorNone(), EvaluatorEField())));
    DataComputer::prepareForRun();
}



void DataComputerEField::computeVector_GPU(bool transferToCPU, uint32_t groupTag) {
    gpuBuffer.d_data.memset(0);

    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly, gpd.xs(activeIdx), (float4 *) gpuBuffer.getDevData(),
                  neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                  state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU,
                  neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), cutoff, 0, state->nThreadPerBlock, state->nThreadPerAtom);



}
void DataComputerEField::computeVector_CPU() {
}
void DataComputerEField::appendVector(boost::python::list &) {
}
