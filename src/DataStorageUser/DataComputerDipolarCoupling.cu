/*
namespace MD_ENGINE {
    class DataComputerDipolarCoupling : public DataComputer {
        public:

            void computeScalar_GPU(bool, uint32_t);
            void computeVector_GPU(bool, uint32_t){};
            void computeTensor_GPU(bool, uint32_t){};

            void computeScalar_CPU();
            void computeVector_CPU(){};
            void computeTensor_CPU(){};

            DataComputeDipolarCoupling(State *, std::string computeMode_);
            GPUArrayGlobal<float> sumInvRCubed; //DON'T FORGET THIS MYSTERIOUS K TERM
            uint32_t groupTagB;
            //so these are just length 2 arrays.  First value is used for the result of the sum.  Second value is bit-cast to an int and used to cound how many values are present.

            void appendScalar(boost::python::list &);
            void appendVector(boost::python::list &){};
            void appendTensor(boost::python::list &){};


            void prepareForRun();

    };
};
*/
#include "DataComputerDipolarCoupling.h"
#include "State.h"
#include "PairEvaluatorDipolarCoupling.h"
#include "EvaluatorWrapper.h"
using namespace MD_ENGINE;
using std::cout;
using std::endl;

__global__ void coalesceInvR6(int nAtoms, float4 *fs, float *uncoalesced, int *counter,  float *res, uint32_t groupTagA) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint32_t groupTag = __float_as_uint(fs[idx].w);
        if (groupTag & groupTagA) {
            int writeIdx = atomicAdd(counter, 1);
            res[writeIdx] = uncoalesced[idx];
        }
    }
}

DataComputerDipolarCoupling::DataComputerDipolarCoupling(State *state_, std::string computeMode_, std::string groupHandleA_, std::string groupHandleB_, double magnetoA_, double magnetoB_) : DataComputer(state_, computeMode_, false), groupHandleA(groupHandleA_), groupHandleB(groupHandleB_), magnetoA(magnetoA_), magnetoB(magnetoB_) {
    groupTagA = state->groupTagFromHandle(groupHandleA);
    groupTagB = state->groupTagFromHandle(groupHandleB);
}

void DataComputerDipolarCoupling::computeScalar_GPU(bool transferToHost, uint32_t groupTag) {
    //this can be cast as a group-group energy computation

    //ignoring groupTag argument - will use groupTagA
    int nAtoms = state->atoms.size();
    //using buffer reduce to assign idxs to coalesced memory
    gpuBufferReduce.d_data.memset(0);
    //this will store un-coalesced couplings
    gpuBuffer.d_data.memset(0);
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    //hijacking energy group-group calculation to compute sum of 1/r^3, which we'll then multiple by some coefficient
    evalWrap->energyGroupGroup(nAtoms, nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), gpuBuffer.getDevData(),neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, rCutSqrArray.getDevData() /*giving junk data to the parameters*/, numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), 0, groupTagA, groupTagB, state->nThreadPerBlock, state->nThreadPerAtom);


    coalesceInvR6<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.fs(activeIdx), gpuBuffer.getDevData(), (int *) gpuBufferReduce.getDevData(), coalescedInvR6.getDevData(), groupTagA);
    if (transferToHost) {
        coalescedInvR6.dataToHost();
    }
    //REMEMBER THE 0.7 FROM RYAN'S EMAIL -- no, take care of this on python side
}




void DataComputerDipolarCoupling::computeScalar_CPU() {
    //See Rotational-Echo, Double-Resonance NMR (2008) by Terry Gullion
    //assuming real units
    double muo = 4*M_PI*1e-7;
    double hbar = 6.62607004e-34 / (2*M_PI);

    double coef = pow(muo * magnetoA * magnetoB * hbar / (8 * M_PI * M_PI), 2);
    for (int i=0; i<coalescedInvR6.h_data.size(); i++) {
        double invr6 = coalescedInvR6.h_data[i]; //this is in 1/ang^3 right now
        double invr6SI = invr6 * 1e60; //now in 1/m^6 - I hope the numerical precision is good enough
        couplingsSqr[i] = coef * invr6SI ; //normally /4pi, but I am converting to hz from rad/sec
    }
}

void DataComputerDipolarCoupling::appendScalar(boost::python::list &vals) {
    vals.append(couplingsSqr);
}

void DataComputerDipolarCoupling::prepareForRun() {

    int nTypes = state->atomParams.numTypes;
    double rCutSqr = state->rCut*state->rCut;
    rCutSqrArray = GPUArrayGlobal<float>(nTypes*nTypes);
    for (int i=0; i<nTypes*nTypes; i++) {
        rCutSqrArray.h_data[i] = rCutSqr;
    }
    rCutSqrArray.dataToDevice();
    int nInGroup = state->countNumInGroup(groupTagA);
    coalescedInvR6 = GPUArrayGlobal<float>(nInGroup);
    couplingsSqr = std::vector<double>(nInGroup);
    EvaluatorDipolarCoupling eval;
    evalWrap = pickEvaluator<EvaluatorDipolarCoupling, 1, true>(eval, nullptr);
    DataComputer::prepareForRun(); //gpuBuffer member will be of correct size


}
