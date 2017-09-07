#include "IntegratorVerlet.h"

#include <chrono>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "Logging.h"
#include "State.h"
#include "Fix.h"
#include "cutils_func.h"

using namespace MD_ENGINE;
using std::cout;
using std::endl;

namespace py = boost::python;

__global__ void nve_v_cu(int nAtoms, float4 *vs, float4 *fs, float dtf) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = dtf * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

__global__ void nve_x_cu(int nAtoms, float4 *xs, float4 *vs, float dt) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update position by a full timestep
        float4 vel = vs[idx];
        float4 pos = xs[idx];

        //printf("pos %f %f %f\n", pos.x, pos.y, pos.z);
        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        float3 dx = dt*make_float3(vel);
        pos += dx;
        xs[idx] = pos;
    }
}

__global__ void nve_xPIMD_cu(int nAtoms, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, BoundsGPU bounds, float dt) {

   
}

//so preForce_cu is split into two steps (nve_v, nve_x) if any of the fixes (barostat, for example), need to throw a step in there (as determined by requiresPostNVE_V flag)
__global__ void preForce_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs,
                            float dt, float dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = dtf * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;

        // Update position by a full timestep
        float4 pos = xs[idx];

        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        float3 dx = dt*make_float3(vel);
        pos += dx;
        xs[idx] = pos;

        // Set forces to zero before force calculation
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

// alternative version of preForce_cu which allows for normal-mode propagation of RP dynamics
// need to pass nPerRingPoly and omega_P
__global__ void preForcePIMD_cu(int nAtoms, int nPerRingPoly, float omegaP, float4 *xs, float4 *vs, float4 *fs, BoundsGPU bounds,
                            float dt, float dtf)
{
}
    //if (useThread && amRoot ) {
    //    printf("--xx = %f\n",xs[idx].x);
    //    printf("--vx = %f\n",vs[idx].x);
    //    printf("--fx = %f\n",fs[idx].x);
    //    printf("R = np.array([");
    //    for (int i = 0; i <nPerRingPoly; i++) {
    //        printf("%f, ",xs[threadIdx.x+i].x);
    //    }
    //    printf("])\n");
    //    printf("V = np.array([");
    //    for (int i = 0; i <nPerRingPoly; i++) {
    //        printf("%f, ",vs[threadIdx.x+i].x);
    //    }
    //    printf("])\n");
    //}

__global__ void postForce_cu(int nAtoms, float4 *vs, float4 *fs, float dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocities by a halftimestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 dv = dtf * invmass * make_float3(force);
        vel += dv;
        vs[idx] = vel;
    }
}

IntegratorVerlet::IntegratorVerlet(State *state_)
    : Integrator(state_)
{

}

void IntegratorVerlet::setInterpolator() {
    for (Fix *f: state->fixes) {
        if ( f->isThermostat && f->groupHandle == "all" ) {
            std::string t = "temp";
            tempInterpolator = f->getInterpolator(t);
            return;
        }
    }
    mdError("No thermostat found when setting up PIMD");
}


double IntegratorVerlet::run(int numTurns)
{

    basicPreRunChecks();
    //basicPrepare(numTurns); //nlist built here
    //force(false);

    std::vector<bool> prepared = basicPrepare(numTurns);
    force(true);

    for (int i = 0; i<prepared.size(); i++) {
        if (!prepared[i]) {
            for (Fix *f : state->fixes) {
                bool isPrepared = f->prepareForRun();
                if (!isPrepared) {
                    mdError("A fix is unable to be instantiated correctly.");
                }
            }
        }
    }
    if (state->nPerRingPoly>1) {
        setInterpolator();
    }
    int periodicInterval = state->periodicInterval;

	
    auto start = std::chrono::high_resolution_clock::now();

    DataManager &dataManager = state->dataManager;
    dtf = 0.5f * state->dt * state->units.ftm_to_v;
    int tuneEvery = state->tuneEvery;
    bool haveTunedWithData = false;
    double timeTune = 0;
    for (int i=0; i<numTurns; ++i) {

        if (state->turn % periodicInterval == 0 or state->turn == state->nextForceBuild) {
            state->gridGPU.periodicBoundaryConditions();
        }

        int virialMode = dataManager.getVirialModeForTurn(state->turn);

        stepInit(virialMode==1 or virialMode==2);

        // Perform first half of velocity-Verlet step
        if (state->requiresPostNVE_V) {
            nve_v();
            postNVE_V();
            nve_x();
        } else {
            preForce();
        }
        postNVE_X();
        //printf("preForce IS COMMENTED OUT\n");

        handleBoundsChange();

        if ((state->turn-state->runInit) % tuneEvery == 0 and state->turn > state->runInit) {
            //this goes here because forces are zero at this point.  I don't need to save any forces this way
            timeTune += tune();
        } else if (not haveTunedWithData and state->turn-state->runInit < tuneEvery and state->nlistBuildCount > 20) {
            timeTune += tune();
            haveTunedWithData = true;
        }

        // Recalculate forces
        force(virialMode);

        //quits if ctrl+c has been pressed
        checkQuit();


        // Perform second half of velocity-Verlet step
        postForce();

        stepFinal();

        //HEY - MAKE DATA APPENDING HAPPEN WHILE SOMETHING IS GOING ON THE GPU.  
        doDataComputation();
        doDataAppending();
        dataManager.clearVirialTurn(state->turn);
        asyncOperations();

        //! \todo The following parts could also be moved into stepFinal
        state->turn++;
        if (state->verbose && (i+1 == numTurns || state->turn % state->shoutEvery == 0)) {
            mdMessage("Turn %d %.2f percent done.\n", (int)state->turn, 100.0*(i+1)/numTurns);
        }
    }

    //! \todo These parts could be moved to basicFinish()
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("after run\n");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double ptsps = state->atoms.size()*numTurns / (duration.count() - timeTune);
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), ptsps);

    basicFinish();
    return ptsps;
}

void IntegratorVerlet::nve_v() {
    uint activeIdx = state->gpd.activeIdx();
    nve_v_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            dtf);
}

void IntegratorVerlet::nve_x() {
    uint activeIdx = state->gpd.activeIdx();
    if (state->nPerRingPoly == 1) {
    	nve_x_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
    	        state->atoms.size(),
    	        state->gpd.xs.getDevData(),
    	        state->gpd.vs.getDevData(),
    	        state->dt); }
    else {
	    // get target temperature from thermostat fix
	    double temp = tempInterpolator->getCurrentVal();
	    int   nPerRingPoly = state->nPerRingPoly;
        int   nRingPoly = state->atoms.size() / nPerRingPoly;
	    float omegaP    = (float) state->units.boltz * temp / state->units.hbar  ;
    	nve_xPIMD_cu<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(float3) * 3 *PERBLOCK>>>(
	        state->atoms.size(),
	 	    nPerRingPoly,
		    omegaP,
    	    state->gpd.xs.getDevData(),
    	    state->gpd.vs.getDevData(),
            state->boundsGPU,
    	    state->dt); 
    }
}
void IntegratorVerlet::preForce()
{
    uint activeIdx = state->gpd.activeIdx();
    if (state->nPerRingPoly == 1) {
    	preForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
    	        state->atoms.size(),
    	        state->gpd.xs.getDevData(),
    	        state->gpd.vs.getDevData(),
    	        state->gpd.fs.getDevData(),
    	        state->dt,
    	        dtf); }
    else {
    
	    // get target temperature from thermostat fix
	    // XXX: need to think about how to handle if no thermostat
	    double temp = tempInterpolator->getCurrentVal();
        
	    int   nPerRingPoly = state->nPerRingPoly;
        int   nRingPoly    = state->atoms.size() / nPerRingPoly;
	    float omegaP       = (float) state->units.boltz * temp / state->units.hbar ;
   
        // called on a per bead basis
        preForcePIMD_cu<<<NBLOCK(state->atoms.size()), PERBLOCK, sizeof(float3) * 3 *PERBLOCK >>>(
	        state->atoms.size(),
	     	nPerRingPoly,
	    	omegaP,
        	state->gpd.xs.getDevData(),
        	state->gpd.vs.getDevData(),
        	state->gpd.fs.getDevData(),
            state->boundsGPU,
        	state->dt,
        	dtf ); 
    }
}

void IntegratorVerlet::postForce()
{
    uint activeIdx = state->gpd.activeIdx();
    postForce_cu<<<NBLOCK(state->atoms.size()), PERBLOCK>>>(
            state->atoms.size(),
            state->gpd.vs.getDevData(),
            state->gpd.fs.getDevData(),
            dtf);
}

void export_IntegratorVerlet()
{
    py::class_<IntegratorVerlet,
               boost::shared_ptr<IntegratorVerlet>,
               py::bases<Integrator>,
               boost::noncopyable>
    (
        "IntegratorVerlet",
        py::init<State *>()
    )
    .def("run", &IntegratorVerlet::run,(py::arg("numTurns")))
    ;
}
