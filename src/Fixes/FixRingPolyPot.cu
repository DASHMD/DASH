#include "FixRingPolyPot.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"

const std::string RingPolyPotType = "RingPolyPot";
using namespace std;
namespace py = boost::python;

// the constructor for FixRingPolyPot
FixRingPolyPot::FixRingPolyPot(SHARED(State) state_, std::string handle_, std::string groupHandle_)
  : Fix(state_, handle_, groupHandle_, RingPolyPotType, true, false,false, 1 ) { };

void __global__ compute_RP_energy_cu(int nAtoms, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, float4 *fs, BoundsGPU bounds, float *perParticleEng, uint groupTag, float mvv_to_eng ) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        uint groupTagAtom = * (uint *) &fs[idx].w;
        // Check if atom is part of the group
        if (groupTagAtom & groupTag) {
            float mi    = (float) 1.0 / vs[idx].w;
            int beadIdx = idx% nPerRingPoly; // time slice
            int beadIdp = (beadIdx + 1) % nPerRingPoly;
            float3 ri   = make_float3(xs[idx]);                     // position at i
            float3 rip1 = make_float3(xs[idx+beadIdp-beadIdx]);     // position at i+1
            float3 dr   = bounds.minImage(ri - rip1);               // difference vector b/w i, i+1
            float  dr2  = lengthSqr(dr);                            // difference vector dot product
            float  eng  = 0.5 * mi * omegaP * omegaP * dr2;         // 0.5*mi*omegaP^2*(ri-rip1)^2 
                                                                    // note: each atom is assigned one bond
            perParticleEng[idx] += eng * mvv_to_eng;                // add energy contribution
        }
    }
    
}

bool FixRingPolyPot::prepareForRun() {
    Fix::prepareForRun();
    prepared = true;
    return prepared;
}

void FixRingPolyPot::singlePointEng(float *perParticleEng) {
        GPUData &gpd    = state->gpd;
        int activeIdx   = gpd.activeIdx();
        int nAtoms      = state->atoms.size();
        int nPerRingPoly= state->nPerRingPoly;
        double temp;
        for (Fix *f: state->fixes) {
            if (f->isThermostat && f->groupHandle == "all" ) {
                std::string t = "temp";
                temp = f->getInterpolator(t)->getCurrentVal();
            }
        }
        float omegaP = (float) state->units.boltz * temp / state->units.hbar  ;
        compute_RP_energy_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, nPerRingPoly,omegaP,
                gpd.xs(activeIdx),gpd.vs(activeIdx),gpd.fs(activeIdx),state->boundsGPU,perParticleEng, groupTag,state->units.mvv_to_eng);
};

// export function
void export_FixRingPolyPot() {
	py::class_<FixRingPolyPot, SHARED(FixRingPolyPot), py::bases<Fix>, boost::noncopyable > (
		"FixRingPolyPot",
		py::init<SHARED(State), string, string> (
			py::args("state", "handle", "groupHandle")
		)
	)
	;
}


