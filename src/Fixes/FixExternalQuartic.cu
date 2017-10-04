#include "FixExternalQuartic.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "ExternalEvaluate.h"

const std::string ExternalQuarticType = "ExternalQuartic";
using namespace std;
namespace py = boost::python;

// the constructor for FixExternalQuartic
FixExternalQuartic::FixExternalQuartic(SHARED(State) state_, std::string handle_, std::string groupHandle_,
                                 Vector k1_, Vector k2_, Vector k3_, Vector k4_ , Vector r0_)
  : FixExternal(state_, handle_, groupHandle_, ExternalQuarticType, true,  false, 1 ),
    k1(k1_.asFloat3()), k2(k2_.asFloat3()), k3(k3_.asFloat3()), k4(k4_.asFloat3()),
    r0(r0_.asFloat3()) { };

// compute function
void FixExternalQuartic::compute(int virialMode) {
	GPUData &gpd  = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n         = state->atoms.size();
	if (virialMode==2 or virialMode==1) {
		compute_force_external<EvaluatorExternalQuartic, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), groupTag, gpd.virials.d_data.data(), evaluator);
	} else {
		compute_force_external<EvaluatorExternalQuartic, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), groupTag, gpd.virials.d_data.data(), evaluator);
	}
};

void FixExternalQuartic::singlePointEng(float *perParticleEng) {
        GPUData &gpd  = state->gpd;
        int activeIdx = gpd.activeIdx();
        int n         = state->atoms.size();
        compute_energy_external<EvaluatorExternalQuartic> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), perParticleEng, groupTag, evaluator);
};


bool FixExternalQuartic::prepareForRun() {
    // set this fix's evaulator with the appropriate parameters
    evaluator = EvaluatorExternalQuartic(k1,k2,k3,k4,r0);
    prepared = true;
    return prepared;
};

bool FixExternalQuartic::postRun () {
    return true;
};

// export function
void export_FixExternalQuartic() {
	py::class_<FixExternalQuartic, SHARED(FixExternalQuartic), py::bases<FixExternal>, boost::noncopyable > (
		"FixExternalQuartic",
		py::init<SHARED(State), string, string, Vector, Vector, Vector, Vector, Vector> (
			py::args("state", "handle", "groupHandle", "k1","k2","k3","k4", "r0")
		)
	)
	.def_readwrite("k1", &FixExternalQuartic::k1)
	.def_readwrite("k2", &FixExternalQuartic::k2)
	.def_readwrite("k3", &FixExternalQuartic::k3)
	.def_readwrite("k4", &FixExternalQuartic::k4)
	.def_readwrite("r0", &FixExternalQuartic::r0)
	;
}


