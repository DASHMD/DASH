#include "IntegratorGradientDescent.h"

#include <chrono>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "Logging.h"
#include "State.h"
#include "cutils_func.h"
using namespace MD_ENGINE;

namespace py = boost::python;

__global__ void step_cu(int nAtoms, float4 *xs, float4 *vs, float4 *fs, float dt, float dtf)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        // Update velocity by a half timestep
        float4 vel = vs[idx];
        float invmass = vel.w;

        float4 force = fs[idx];

        float3 v = dtf * invmass * make_float3(force);

        // Update position by a full timestep
        float4 pos = xs[idx];

        //printf("vel %f %f %f\n", vel.x, vel.y, vel.z);
        float3 dx = dt*v;
        pos += dx;
        xs[idx] = pos;

        // Set forces to zero before force calculation
        fs[idx] = make_float4(0.0f, 0.0f, 0.0f, force.w);
    }
}

void IntegratorGradientDescent::step(double coef) {
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    double dtf = state->dt * state->units.ftm_to_v * coef;
    double dt = state->dt * coef;
    step_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.xs.getDevData(), gpd.vs.getDevData(), gpd.fs.getDevData(), dt, dtf);
}

IntegratorGradientDescent::IntegratorGradientDescent(State *state_)
    : Integrator(state_)
{

}

double IntegratorGradientDescent::getSumForceSqr() {
    forceSingle(false);
    GPUArrayGlobal<float> force(1);
    force.memsetByVal(0);
    int nAtoms = state->atoms.size();
    int warpSize = state->devManager.prop.warpSize;
    accumulate_gpu<float, float4, SumVectorSqr3D, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>> 
        (
         force.getDevData(),
         state->gpd.fs.getDevData(),
         nAtoms,
         warpSize,
         SumVectorSqr3D()
        );
    force.dataToHost();
    cudaDeviceSynchronize();
    return force.h_data[0];
}

void IntegratorGradientDescent::run(int numTurns, double coef)
{

    basicPreRunChecks();

    // basicPrepare now only handles State prepare and sending global State data to device
    basicPrepare(numTurns);

    // prepare the fixes that do not require forces to be computed
    prepareFixes(false);
    
    forceInitial(true);

    // prepare the fixes that require forces to be computed on instantiation;
    prepareFixes(true);

    // finally, prepare barostats, thermostats, datacomputers, etc.
    // datacomputers are prepared first, then the barostats, thermostats, etc.
    // prior to datacomputers being prepared, we iterate over State, and the groups in simulation 
    // collect their NDF associated with their group
    prepareFinal();

    verifyPrepared();

    int periodicInterval = state->periodicInterval;

    auto start = std::chrono::high_resolution_clock::now();
    DataManager &dataManager = state->dataManager;
    for (int i=0; i<numTurns; ++i) {
        if (state->turn % periodicInterval == 0) {
            state->gridGPU.periodicBoundaryConditions();
        }

        int virialMode = dataManager.getVirialModeForTurn(state->turn);
        stepInit(virialMode==1 or virialMode==2);

        // Calculate forces
        forceSingle(virialMode);

        //quits if ctrl+c has been pressed
        checkQuit();

        // Descend along gradient
        step(coef);

        stepFinal();

        asyncOperations();
        doDataComputation();
        //HEY - MAKE DATA APPENDING HAPPEN WHILE SOMETHING IS GOING ON THE GPU.  
        doDataAppending();
        dataManager.clearVirialTurn(state->turn);

        //! \todo The following parts could also be moved into stepFinal
        state->turn++;
        if (state->verbose && (i+1 == numTurns || state->turn % state->shoutEvery == 0)) {
            mdMessage("Turn %d %.2f percent done.\n", (int)state->turn, 100.0*(i+1)/numTurns);
        }
    }
    double sumForceSqr = getSumForceSqr();



    //! \todo These parts could be moved to basicFinish()
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    mdMessage("runtime %f\n%e particle timesteps per second\n",
              duration.count(), state->atoms.size()*numTurns / duration.count());
    mdMessage("Total force %f average force %f\n", sqrt(sumForceSqr), sqrt(sumForceSqr)/state->atoms.size());

    basicFinish();
}



void export_IntegratorGradientDescent()
{
    py::class_<IntegratorGradientDescent,
               boost::shared_ptr<IntegratorGradientDescent>,
               py::bases<Integrator>,
               boost::noncopyable>
    (
        "IntegratorGradientDescent",
        py::init<State *>()
    )
    .def("run", &IntegratorGradientDescent::run)
    ;
}
