#include "FixLangevin.h"
#define INVALID_VAL INT_MAX
#include "cutils_math.h"

#include "State.h"
const std::string LangevinType = "Langevin";
namespace py = boost::python;



__global__ void compute_cu(int nAtoms, float4 *vs, float4 *fs, curandState_t *randStates, float dt, float T, float gamma, float boltz, float mvv_to_e, float ftm_to_v, bool useMass) {

    int idx = GETIDX();
    if (idx < nAtoms) {

        curandState_t *randState = randStates + idx;
        curandState_t localState=*randState;
        float3 Wiener;
        Wiener.x=curand_uniform(&localState)-0.5f;
        Wiener.y=curand_uniform(&localState)-0.5f;
        Wiener.z=curand_uniform(&localState)-0.5f;
        *randState=localState;
        float4 vel_whole = vs[idx];
        float3 vel = make_float3(vel_whole);

        float invMass = vel_whole.w;
        if (!useMass) {
            invMass = 1.0f;
        }
        float dragFactor = gamma / (invMass * ftm_to_v);
        float kickFactor = sqrtf((24.0f * boltz * gamma * T ) / (invMass * dt * mvv_to_e)) / ftm_to_v;

        float4 force = fs[idx];


        float3 dForce = Wiener * kickFactor - vel * dragFactor;

         
        force += dForce;
        fs[idx]=force;
    }
}







void FixLangevin::setDefaults() {
    seed = 0;
    gamma = 1.0;
}

FixLangevin::FixLangevin(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_, double temp_) : Interpolator(temp_), Fix(state_, handle_, groupHandle_, LangevinType, false, false, false, 1) {
    setDefaults();
}

FixLangevin::FixLangevin(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_, py::list intervals_, py::list temps_) : Interpolator(intervals_, temps_), Fix(state_, handle_, groupHandle_, LangevinType, false, false, false, 1) {
    setDefaults();
}

FixLangevin::FixLangevin(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_, py::object tempFunc_) : Interpolator(tempFunc_), Fix(state_, handle_, groupHandle_, LangevinType, false, false, false, 1) {
    setDefaults();
}

void __global__ initRandStates(int nAtoms, curandState_t *states, int seed,int turn) {
    int idx = GETIDX();
    curand_init(seed, idx, turn, states + idx);

}


bool FixLangevin::prepareForRun() {
    turnBeginRun = state->runInit;
    turnFinishRun = state->runInit + state->runningFor;
    randStates = GPUArrayDeviceGlobal<curandState_t>(state->atoms.size());
    initRandStates<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), randStates.data(), seed,state->turn);
    return true;
}

bool FixLangevin::postRun() {
    finished = true;
    return true;
}

void FixLangevin::setParams(double seed_, double gamma_) {
    if (seed_ != INVALID_VAL) {
        seed = seed_;
    }
    if (gamma_ != INVALID_VAL) {
        gamma = gamma_;
    }
}
void FixLangevin::compute(int virialMode) {
    computeCurrentVal(state->turn);
    double temp = getCurrentVal();
    compute_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(state->atoms.size(), state->gpd.vs.getDevData(), state->gpd.fs.getDevData(), randStates.data(), state->dt, temp, gamma, state->units.boltz, state->units.mvv_to_eng, state->units.ftm_to_v, true);
    
}




void export_FixLangevin() {
    py::class_<FixLangevin, SHARED(FixLangevin), py::bases<Fix> > (
        "FixLangevin", 
        py::init<boost::shared_ptr<State>, std::string, std::string, py::list, py::list>(
            py::args("state", "handle", "groupHandle", "intervals", "temps")
            )

        
    )
   //HEY - ORDER IS IMPORTANT HERE.  LAST CONS ADDED IS CHECKED FIRST. A DOUBLE _CAN_ BE CAST AS A py::object, SO IF YOU PUT THE TEMPFUNC CONS LAST, CALLING WITH DOUBLE AS ARG WILL GO THERE, NOT TO CONST TEMP CONSTRUCTOR 
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, py::object>(
                
            py::args("state", "handle", "groupHandle", "tempFunc")
                )
            )
    .def(py::init<boost::shared_ptr<State>, std::string, std::string, double>(
            py::args("state", "handle", "groupHandle", "temp")
                )
            )
    .def("setParameters", &FixLangevin::setParams, 
         (py::arg("seed") = INVALID_VAL, py::arg("gamma")=INVALID_VAL)
        )
    ;
}
