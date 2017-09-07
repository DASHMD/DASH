#include "FixNVTAndersen.h"

#define INVALID_VAL INT_MAX
#include "Bounds.h"
#include "cutils_func.h"
#include "State.h"

namespace py=boost::python;

using namespace std;

void FixNVTAndersen::setDefaults() {
    seed=0;
}

const std::string NVTAndersenType = "NVTAndersen";

FixNVTAndersen::FixNVTAndersen(SHARED(State) state_, string handle_, string groupHandle_, py::list intervals_, py::list temps_, float nu_, int applyEvery_)
    : Fix(state_, handle_, groupHandle_, NVTAndersenType, false, false, false, applyEvery_),
      Interpolator(intervals_, temps_), 
      tempComputer(state, "scalar")
{
    setDefaults();
    isThermostat = true;
    nudt         = state_->dt * nu_; 
}

FixNVTAndersen::FixNVTAndersen(SHARED(State) state_, string handle_, string groupHandle_, py::object tempFunc_, float nu_, int applyEvery_)
    : Fix(state_, handle_, groupHandle_, NVTAndersenType, false, false, false, applyEvery_),
      Interpolator(tempFunc_), 
      tempComputer(state, "scalar")
{
    setDefaults();
    isThermostat = true;
    nudt         = state_->dt * nu_; 
}

FixNVTAndersen::FixNVTAndersen(SHARED(State) state_, string handle_, string groupHandle_, double constTemp_, float nu_, int applyEvery_)
    : Fix(state_, handle_, groupHandle_, NVTAndersenType, false, false, false, applyEvery_),
      Interpolator(constTemp_), 
      tempComputer(state, "scalar")
{
    setDefaults();
    isThermostat = true;
    nudt         = state_->dt * nu_; 
}

void __global__ initRand(int nAtoms, curandState_t *states, int seed,int turn) {
    int idx = GETIDX();
    curand_init(seed, idx, turn, states + idx);

}

bool FixNVTAndersen::prepareForRun() {
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    tempComputer.prepareForRun();
    randStates = GPUArrayDeviceGlobal<curandState_t>(state->atoms.size());
    initRand<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), randStates.data(), seed,state->turn);
    return true;
}

void __global__ resample_no_tags_cu(int nAtoms, float4 *vs, curandState_t *randStates, float tempSet, float nudt, float boltz, float mvv_to_e) {
    int idx = GETIDX();
    if (tempSet > 0 and idx < nAtoms) {
        curandState_t *randState = randStates + idx;
        curandState_t localState=*randState;
        if ( curand_uniform(&localState) <= nudt ) {
            // resample from Boltzmann distribution
            float4 vnew    = vs[idx];
            float  invmass = vnew.w;
            float  sigma   = sqrtf(boltz * tempSet * invmass / mvv_to_e);
                float sx; float sy; float sz;
                sx = curand_normal(&localState);
                sy = curand_normal(&localState);
                sz = curand_normal(&localState);
                vnew.x = sigma*sx;
                vnew.y = sigma*sy;
                vnew.z = sigma*sz;
            vs[idx]= vnew;
        }
        *randState=localState;
    }
}

void __global__ resample_cu(int nAtoms, uint groupTag, float4 *vs, float4 *fs, curandState_t *randStates, float tempSet, float nudt, float boltz, float mvv_to_e) {

    int idx = GETIDX();
    if (tempSet > 0 and idx < nAtoms) {
        curandState_t *randState = randStates + idx;
        curandState_t localState=*randState;
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            if ( curand_uniform(&localState) <= nudt ) {
                // resample from Boltzmann distribution
                float4 vnew    = vs[idx];
                float  invmass = vnew.w;
                float  sigma   = sqrtf(boltz * tempSet * invmass / mvv_to_e);
                float sx; float sy; float sz;
                sx = curand_normal(&localState);
                sy = curand_normal(&localState);
                sz = curand_normal(&localState);
                vnew.x = sigma*sx;
                vnew.y = sigma*sy;
                vnew.z = sigma*sz;
                vs[idx]= vnew;
            }
            *randState=localState;
        }
    }
}

void FixNVTAndersen::compute(int virialMode) {

    tempComputer.computeScalar_GPU(true, groupTag);
    int nAtoms    = state->atoms.size();
    int64_t turn  = state->turn;
    computeCurrentVal(turn);
    double temp   = getCurrentVal();
    GPUData &gpd  = state->gpd;
    int activeIdx = gpd.activeIdx();

    if (groupTag == 1) {
        resample_no_tags_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.vs(activeIdx), randStates.data(), 
                temp, nudt,state->units.boltz,state->units.mvv_to_eng);

    } else {
        resample_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, groupTag, gpd.vs(activeIdx),gpd.fs(activeIdx), randStates.data(), 
                temp, nudt,state->units.boltz,state->units.mvv_to_eng);
    }
}


bool FixNVTAndersen::postRun() {
    finishRun();
    return true;
}

void FixNVTAndersen::setParams(double seed_) {
    if (seed_ != INVALID_VAL) {
        seed = seed_;
    }
}

Interpolator *FixNVTAndersen::getInterpolator(std::string type) {
    if (type == "temp") {
        return (Interpolator *) this;
    }
    return nullptr;
}


void export_FixNVTAndersen() {
    py::class_<FixNVTAndersen, SHARED(FixNVTAndersen), py::bases<Fix>, boost::noncopyable > (
        "FixNVTAndersen", 
        py::init<boost::shared_ptr<State>, string, string, py::list, py::list, py::optional<float,int> >(
            py::args("state", "handle", "groupHandle", "intervals", "temps","nu", "applyEvery")
            )

        
    )
   
    .def(py::init<boost::shared_ptr<State>, string, string, py::object, py::optional<float,int> >(
                
            py::args("state", "handle", "groupHandle", "tempFunc","nu","applyEvery")
                )
            )
    .def(py::init<boost::shared_ptr<State>, string, string, double, py::optional<float,int> >(
            py::args("state", "handle", "groupHandle", "temp","nu", "applyEvery")
                )
            )
    .def("setParameters", &FixNVTAndersen::setParams,
         (py::arg("seed") = INVALID_VAL)
        )
    ;
}
