#include "FixExternalHarmonic.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "ExternalEvaluate.h"

const std::string ExternalHarmonicType = "ExternalHarmonic";
using namespace std;
namespace py = boost::python;

// the constructor for FixExternalHarmonic
FixExternalHarmonic::FixExternalHarmonic(SHARED(State) state_, std::string handle_, std::string groupHandle_,
                                 Vector k_, Vector r0_)
  : FixExternal(state_, handle_, groupHandle_, ExternalHarmonicType, true,  false, 1 ),
    k(k_.asFloat3()), r0(r0_.asFloat3()) { };

// compute function
void FixExternalHarmonic::compute(int virialMode) {
	GPUData &gpd  = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n         = state->atoms.size();
	if (virialMode==2 or virialMode == 1) {
		compute_force_external<EvaluatorExternalHarmonic, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), groupTag, gpd.virials.d_data.data(), evaluator);
	} else {
		compute_force_external<EvaluatorExternalHarmonic, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), groupTag, gpd.virials.d_data.data(), evaluator);
	}
};

void FixExternalHarmonic::singlePointEng(float *perParticleEng) {
        GPUData &gpd  = state->gpd;
        int activeIdx = gpd.activeIdx();
        int n         = state->atoms.size();
        compute_energy_external<EvaluatorExternalHarmonic> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), perParticleEng, groupTag, evaluator);
};


bool FixExternalHarmonic::prepareForRun() {
    // set this fix's evaulator with the appropriate parameters
    evaluator = EvaluatorExternalHarmonic(k, r0);
    prepared = true;
    return prepared;
};

bool FixExternalHarmonic::postRun () {
    return true;
};

// export function
void export_FixExternalHarmonic() {
	py::class_<FixExternalHarmonic, SHARED(FixExternalHarmonic), py::bases<FixExternal>, boost::noncopyable > (
		"FixExternalHarmonic",
		py::init<SHARED(State), string, string, Vector, Vector> (
			py::args("state", "handle", "groupHandle", "k", "r0")
		)
	)
	.def_readwrite("k", &FixExternalHarmonic::k)
	.def_readwrite("r0", &FixExternalHarmonic::r0)
	;
}


