#include "FixChargePairDSF.h"
#include "BoundsGPU.h"
#include "GPUData.h"
#include "GridGPU.h"
#include "State.h"

#include "boost_for_export.h"
#include "cutils_func.h"
#include "EvaluatorWrapper.h"
#include "PairEvaluatorNone.h"
// #include <cmath>

namespace py=boost::python;
using namespace std;

const std::string chargePairDSFType = "ChargePairDSF";

//Pairwise force shifted damped Coulomb
//Christopher J. Fennel and J. Daniel Gezelter J. Chem. Phys (124), 234104 2006
// Eqn 19.
//force calculation:
//  F=q_i*q_j*[erf(alpha*r    )/r^2   +2*alpha/sqrt(Pi)*exp(-alpha^2*r^2    )/r
//	      -erf(alpha*r_cut)/rcut^2+2*alpha/sqrt(Pi)*exp(-alpha^2*r_cut^2)/r_cut]

//or F=q_i*q_j*[erf(alpha*r    )/r^2   +A*exp(-alpha^2*r^2    )/r- shift*r]
//with   A   = 2*alpha/sqrt(Pi)
//     shift = erf(alpha*r_cut)/r_cut^2+2*alpha/sqrt(Pi)*exp(-alpha^2*r_cut^2)/r_cut


//    compute_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), gpd.qs(activeIdx), alpha, r_cut, A, shift, state->boundsGPU, state->devManager.prop.warpSize, 0.5);// state->devManager.prop.warpSize, sigmas.getDevData(), epsilons.getDevData(), numTypes, state->rCut, state->boundsGPU, oneFourStrength);


FixChargePairDSF::FixChargePairDSF(SHARED(State) state_, string handle_, string groupHandle_) : FixCharge(state_, handle_, groupHandle_, chargePairDSFType, true) {
   setParameters(0.25,9.0);
   canOffloadChargePairCalc = true;
   setEvalWrapper();
};

void FixChargePairDSF::setParameters(float alpha_,float r_cut_)
{
  alpha=alpha_;
  r_cut=r_cut_;
  A= 2.0/sqrt(M_PI)*alpha;
  shift=std::erfc(alpha*r_cut)/(r_cut*r_cut)+A*exp(-alpha*alpha*r_cut*r_cut)/(r_cut);
}

bool FixChargePairDSF::prepareForRun() {
    FixCharge::prepareForRun();

    prepared = true;
    return prepared;
}

std::vector<float> FixChargePairDSF::getRCuts() { 
    std::vector<float> res;
    res.push_back(r_cut);
    return res;
}

void FixChargePairDSF::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                  neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                  state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU,
                  neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), r_cut, virialMode, nThreadPerBlock(), nThreadPerAtom());



}

void FixChargePairDSF::singlePointEng(float * perParticleEng) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng,
                  neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                  state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU,
                  neighborCoefs[0], neighborCoefs[1], neighborCoefs[2],  gpd.qs(activeIdx), r_cut, nThreadPerBlock(), nThreadPerAtom());

}

void FixChargePairDSF::singlePointEngGroupGroup(float * perParticleEng, uint32_t tagA, uint32_t tagB) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energyGroupGroup(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), perParticleEng,
                  neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                  state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU,
                  neighborCoefs[0], neighborCoefs[1], neighborCoefs[2],  gpd.qs(activeIdx), r_cut, tagA, tagB, nThreadPerBlock(), nThreadPerAtom());

}

void FixChargePairDSF::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        if (hasOffloadedChargePairCalc) {
            evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), nullptr); //nParams arg is 1 rather than zero b/c can't have zero sized argument on device
        } else {
            evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), this);
        }
    } else if (evalWrapperMode == "self") {
        evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), this);
    }

}

ChargeEvaluatorDSF FixChargePairDSF::generateEvaluator() {
    return ChargeEvaluatorDSF(alpha, A, shift, state->units.qqr_to_eng,r_cut);
}
void export_FixChargePairDSF() {
    py::class_<FixChargePairDSF, SHARED(FixChargePairDSF), boost::python::bases<FixCharge> > (
        "FixChargePairDSF",
        py::init<SHARED(State), string, string> (
            py::args("state", "handle", "groupHandle"))
    )
    .def("setParameters", &FixChargePairDSF::setParameters,
            (py::arg("alpha"), py::arg("r_cut"))
        )
    ;
}
