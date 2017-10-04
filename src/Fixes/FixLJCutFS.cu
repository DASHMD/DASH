#include "FixLJCutFS.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "list_macro.h"
#include "PairEvaluateIso.h"
#include "State.h"
#include "cutils_func.h"
#include "EvaluatorWrapper.h"

const std::string LJCutType = "LJCutFS";
namespace py = boost::python;

FixLJCutFS::FixLJCutFS(SHARED(State) state_, std::string handle_, std::string mixingRules_)
    : FixPair(state_, handle_, "all", LJCutType, true, false, 1, mixingRules_),
      epsHandle("eps"), sigHandle("sig"), rCutHandle("rCut") {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    initializeParameters("FCutHandle", FCuts);
    paramOrder = {rCutHandle, epsHandle, sigHandle, "FCutHandle"};

    canAcceptChargePairCalc = true;
    setEvalWrapper();
}
void FixLJCutFS::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx),
                      neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                      state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU,
                      neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), chargeRCut, virialMode, nThreadPerBlock(), nThreadPerAtom());



}

void FixLJCutFS::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, nThreadPerBlock(), nThreadPerAtom());


}

void FixLJCutFS::singlePointEngGroupGroup(float *perParticleEng, uint32_t tagA, uint32_t tagB) {
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int numTypes = state->atomParams.numTypes;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    float *neighborCoefs = state->specialNeighborCoefs;

    evalWrap->energyGroupGroup(nAtoms,nPerRingPoly, gpd.xs(activeIdx), gpd.fs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, paramsCoalesced.data(), numTypes, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), chargeRCut, tagA, tagB, nThreadPerBlock(), nThreadPerAtom());

}

bool FixLJCutFS::prepareForRun() {
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
    
	std::function<float(int, int)>fillFCut = [this] (int a, int b) {
        int numTypes = state->atomParams.numTypes;
        float epstimes24=24*squareVectorRef<float>(paramMap[epsHandle]->data(),numTypes,a,b);
        float rCutSqr = pow(squareVectorRef<float>(paramMap[rCutHandle]->data(),numTypes,a,b),2);
        float sig6 = pow(squareVectorRef<float>(paramMap[sigHandle]->data(),numTypes,a,b),6);
        float p1 = epstimes24*2*sig6*sig6;
        float p2 = epstimes24*sig6;
        float r2inv = 1/rCutSqr;
        float r6inv = r2inv*r2inv*r2inv;
        float forceScalar = r6inv * r2inv * (p1 * r6inv - p2)*sqrt(rCutSqr);

        return forceScalar;
    };
    prepareParameters(epsHandle, fillGeo, processEps, false);
	if (mixingRules==ARITHMETICTYPE) {
		prepareParameters(sigHandle, fillArith, processSig, false);
	} else {
		prepareParameters(sigHandle, fillGeo, processSig, false);
	}
    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);


    prepareParameters(rCutHandle, fillRCut, processRCut, true, fillRCutDiag);
    prepareParameters("FCutHandle", fillFCut);
    sendAllToDevice();
    setEvalWrapper();
    prepared = true;
    return prepared;
}

void FixLJCutFS::setEvalWrapper() {
    if (evalWrapperMode == "orig") {
        EvaluatorLJFS eval;
        evalWrap = pickEvaluator<EvaluatorLJFS, 3, true>(eval, chargeCalcFix);
    } else if (evalWrapperMode == "self") {
        EvaluatorLJFS eval;
        evalWrap = pickEvaluator<EvaluatorLJFS, 3, true>(eval, nullptr);
    }
}

std::string FixLJCutFS::restartChunk(std::string format) {
    std::stringstream ss;
    ss << restartChunkPairParams(format);
    return ss.str();
}


bool FixLJCutFS::postRun() {

    return true;
}

void FixLJCutFS::addSpecies(std::string handle) {
    initializeParameters(epsHandle, epsilons);
    initializeParameters(sigHandle, sigmas);
    initializeParameters(rCutHandle, rCuts);
    initializeParameters(rCutHandle, FCuts);

}

std::vector<float> FixLJCutFS::getRCuts() {
    std::vector<float> res;
    std::vector<float> &src = *(paramMap[rCutHandle]);
    for (float x : src) {
        if (x == DEFAULT_FILL) {
            res.push_back(-1);
        } else {
            res.push_back(x);
        }
    }

    return res;
}

void export_FixLJCutFS() {
    py::class_<FixLJCutFS,
                          SHARED(FixLJCutFS),
                          py::bases<FixPair>, boost::noncopyable > (
        "FixLJCutFS",
        py::init<SHARED(State), std::string, py::optional<std::string> > (
            py::args("state", "handle", "mixingRules"))
    );

}
