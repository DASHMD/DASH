#include "FixWallLJ126.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "WallEvaluate.h"

const std::string wallLJ126Type = "WallLJ126";
using namespace std;
namespace py = boost::python;

// the constructor for FixWallLJ126
FixWallLJ126::FixWallLJ126(SHARED(State) state_, std::string handle_, std::string groupHandle_,
                                 Vector origin_, Vector forceDir_, float dist_, float sigma_, float epsilon_)
  : FixWall(state_, handle_, groupHandle_, wallLJ126Type, true,  false, 1, origin_, forceDir_.normalized()),
    dist(dist_), sigma(sigma_), epsilon(epsilon_)
{
    assert(dist >= 0);
};



// this refers to the template in the /Evaluators/ folder - 
void FixWallLJ126::compute(int virialMode) {
	GPUData &gpd = state->gpd;
	int activeIdx = gpd.activeIdx();
	int n = state->atoms.size();
	if (virialMode) {
		compute_wall_iso<EvaluatorWallLJ126, true> <<<NBLOCK(n), PERBLOCK>>>(n,  gpd.xs(activeIdx),
                    gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(),  groupTag, evaluator);
	} else {
		compute_wall_iso<EvaluatorWallLJ126, false> <<<NBLOCK(n), PERBLOCK>>>(n, gpd.xs(activeIdx),
                    gpd.fs(activeIdx), origin.asFloat3(), forceDir.asFloat3(),  groupTag, evaluator);
	}
};

void FixWallLJ126::singlePointEng(float *perParticleEng) {

};



bool FixWallLJ126::prepareForRun() {
    // instantiate this fix's evaulator with the appropriate parameters
    evaluator = EvaluatorWallLJ126(sigma, epsilon, dist);

    return true;
};

bool FixWallLJ126::postRun () {
    return true;
};

// export function
void export_FixWallLJ126() {
	py::class_<FixWallLJ126, SHARED(FixWallLJ126), py::bases<FixWall>, boost::noncopyable > (
		"FixWallLJ126",
		py::init<SHARED(State), string, string, Vector, Vector, float, float, float> (
			py::args("state", "handle", "groupHandle", "origin", "forceDir", "dist", "sigma", "epsilon")
		)
	)
	.def_readwrite("sigma", &FixWallLJ126::sigma)
    .def_readwrite("epsilon", &FixWallLJ126::epsilon)
	.def_readwrite("dist", &FixWallLJ126::dist)
	.def_readwrite("forceDir", &FixWallLJ126::forceDir)
	.def_readwrite("origin", &FixWallLJ126::origin)
	;
}
