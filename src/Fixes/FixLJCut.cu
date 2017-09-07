#include "FixLJCut.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "State.h"
#include "cutils_func.h"
#include "ReadConfig.h"
#include "EvaluatorWrapper.h"
#include "PairEvaluatorLJ.h"
#include "EvaluatorWrapper.h"
//#include "ChargeEvaluatorEwald.h"
using namespace std;
namespace py = boost::python;
const string LJCutType = "LJCut";



FixLJCut::FixLJCut(boost::shared_ptr<State> state_, string handle_, string mixingRules_)
    : FixPair(state_, handle_, "all", LJCutType, true, false, 1, mixingRules_),
    epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut")
{

    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle};
    readFromRestart();
    canAcceptChargePairCalc = true;
    setEvalWrapper();
}

void FixLJCut::compute(int virialMode) {
    int nAtoms       = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms, nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, virialMode, nThreadPerBlock(), nThreadPerAtom());

}

void FixLJCut::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms, nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, nThreadPerBlock(), nThreadPerAtom());
}

void FixLJCut::singlePointEngGroupGroup(float *perParticleEng, uint32_t tagA, uint32_t tagB) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energyGroupGroup(nAtoms, nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, tagA, tagB, nThreadPerBlock(), nThreadPerAtom());
}

void FixLJCut::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        EvaluatorLJ eval;
        evalWrap = pickEvaluator<EvaluatorLJ, 3, true>(eval, chargeCalcFix);
    } else if (evalWrapperMode == "self") {
        EvaluatorLJ eval;
        evalWrap = pickEvaluator<EvaluatorLJ, 3, true>(eval, nullptr);
    }
}

bool FixLJCut::prepareForRun() {
    //loop through all params and fill with appropriate lambda function, then send all to device
    auto fillGeo = [] (float a, float b) {
        return sqrt(a*b);
    };

    auto fillArith = [] (float a, float b) {
        return (a+b) / 2.0;
    };
    auto fillRCut = [this] (float a, float b) {
        return (float) std::fmax(a, b);
    };
    auto none = [] (float a){};

    auto fillRCutDiag = [this] () {
        return (float) state->rCut;
    };

    auto processEps = [] (float a) {
        return 24*a;
    };
    auto processSig = [] (float a) {
        return pow(a, 6);
    };
    auto processRCut = [] (float a) {
        return a*a;
    };
    prepareParameters(epsHandle, fillGeo, processEps, false);
	if (mixingRules==ARITHMETICTYPE) {
		prepareParameters(sigHandle, fillArith, processSig, false);
	} else {
		prepareParameters(sigHandle, fillGeo, processSig, false);
	}
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);

    sendAllToDevice();
    setEvalWrapper();
    return true;
}

string FixLJCut::restartChunk(string format) {
    stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}


bool FixLJCut::postRun() {

    return true;
}

void FixLJCut::addSpecies(string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);

}

vector<float> FixLJCut::getRCuts() { 
    vector<float> res;
    vector<float> &src = *(paramMap[rCutHandle]);
    for (float x : src) {
        if (x == DEFAULT_FILL) {
            res.push_back(-1);
        } else {
            res.push_back(x);
        }
    }

    return res;
}

void export_FixLJCut() {
    py::class_<FixLJCut, boost::shared_ptr<FixLJCut>, py::bases<FixPair>, boost::noncopyable > (
        "FixLJCut",
        py::init<boost::shared_ptr<State>, string, py::optional<string> > (py::args("state", "handle", "mixingRules"))
    )
      ;

}
