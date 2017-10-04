#include "FixNoseHoover.h"
#include <cmath>
#include <string>

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include "cutils_func.h"
#include "cutils_math.h"
#include "Logging.h"
#include "State.h"
#include "Mod.h"

enum PRESSMODE {ISO, ANISO};
enum COUPLESTYLE {NONE, XYZ};
namespace py = boost::python;

std::string NoseHooverType = "NoseHoover";



#define n_ys_5 {

// CUDA function to calculate the total kinetic energy

// CUDA function to rescale particle velocities
__global__ void rescale_cu(int nAtoms, uint groupTag, float4 *vs, float4 *fs, float3 scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            float4 vel = vs[idx];
            vel.x *= scale.x;
            vel.y *= scale.y;
            vel.z *= scale.z;
            vs[idx] = vel;
        }
    }
}

__global__ void rescale_no_tags_cu(int nAtoms, float4 *vs, float3 scale)
{
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 vel = vs[idx];
        vel.x *= scale.x;
        vel.y *= scale.y;
        vel.z *= scale.z;
        vs[idx] = vel;
    }
}


__global__ void barostat_vel_cu(int nAtoms,uint groupTag, float4 *vs,
                                        float4 *fs, float3 addScale,
                                        float3 multScale, float dtf) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        uint groupTagAtom = ((uint *) (fs+idx))[3];
        if (groupTag & groupTagAtom) {
            float4 vel = vs[idx];
            float invmass = vel.w;
            float4 force = fs[idx];
            float3 v = make_float3(vel);
            v *= multScale;
            // apply the additive scaling (additive w.r.t. vel)
            float3 dv = dtf * invmass * make_float3(force) * addScale;
            v += dv;

            // apply the multiplicative scaling to the aggregated quantity
            v *= multScale;
            
            float4 newV = make_float4(v.x, v.y, v.z, invmass);
            vs[idx] = newV;
        }
    }
}

__global__ void barostat_vel_no_tags_cu(int nAtoms, float4 *vs, 
                                float4 *fs, float3 addScale, float3 multScale, float dtf) {

    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 vel = vs[idx];
        float invmass = vel.w;
        float4 force = fs[idx];
        float3 v = make_float3(vel);

        v *= multScale;

        // apply the additive scaling (additive w.r.t. vel)
        float3 dv = dtf * invmass * make_float3(force) * addScale;
        v += dv;

        // apply the multiplicative scaling to the aggregated quantity
        v *= multScale;
        
        float4 newV = make_float4(v.x, v.y, v.z, invmass);
        vs[idx] = newV;
    }
}

// general constructor; may be a thermostat, or a barostat-thermostat
FixNoseHoover::FixNoseHoover(boost::shared_ptr<State> state_, std::string handle_,
                             std::string groupHandle_)
        : 
          Fix(state_,
              handle_,           // Fix handle
              groupHandle_,      // Group handle
              NoseHooverType,   // Fix name
              false,            // forceSingle
              false,
              false,            // requiresCharges
              1,                 // applyEvery
              50                // orderPreference
             ), 
                kineticEnergy(GPUArrayGlobal<float>(2)),
                ke_current(0.0), ndf(0),
                chainLength(20), nTimesteps(1), n_ys(1),
                pchainLength(20),
                nTimesteps_b(1), n_ys_b(1),
                weight(std::vector<double>(n_ys,1.0)),
                //thermPos(std::vector<double>(chainLength,0.0)),
                thermVel(std::vector<double>(chainLength,0.0)),
                thermForce(std::vector<double>(chainLength,0.0)),
                thermMass(std::vector<double>(chainLength,0.0)),
                scale(make_float3(1.0f, 1.0f, 1.0f)),
                pressFreq(6, 0),
                pFlags(6, false),
                tempComputer(state, "scalar"), 
                pressComputer(state, "scalar"), 
                pressMode(PRESSMODE::ISO),
                couple(COUPLESTYLE::XYZ),
                thermostatting(true),
                barostatting(false)
{
    pressComputer.usingExternalTemperature = true;

    // set flag 'requiresPostNVE_V' to true
    requiresPostNVE_V = true;

    // this is a thermostat (if we are barostatting, we are also thermostatting)
    isThermostat = true;

    // denote whether or not this is the first time prepareForRun was called
    // --- need this, because we need to initialize this with proper virials
    //firstPrepareCalled = true;

}


void FixNoseHoover::parseKeyword(std::string keyword) {
    if (keyword == "ISO") {
        pressMode = PRESSMODE::ISO;
        couple = COUPLESTYLE::XYZ;
        // state of the pressComputer defaults to scalar (see the constructor above)
    } else if (keyword == "ANISO") {
        // allow the x, y, z dimensions to vary dynamically according to their instantaneous stress
        // --- so, still restricting to the hydrostatic pressure (1/3 * \sigma_{ii}) but
        //     not coupling.
        pressMode = PRESSMODE::ANISO;
        couple = COUPLESTYLE::NONE;
        // change the state of our pressComputer - and tempComputer - to "tensor"
        pressComputer = MD_ENGINE::DataComputerPressure(state,"tensor");
        tempComputer  = MD_ENGINE::DataComputerTemperature(state,"tensor");
        // and assert again that we are using an external temperature computer (tempComputer) for
        // our pressure computer
        pressComputer.usingExternalTemperature = true;

    } else {
        barostatErrorMessage = "Invalid keyword in FixNoseHoover::setPressure():\n";
        barostatErrorMessage += "Valid options are \"ISO\", \"ANISO\";";
        mdError("See above error message");

    }

    // regulating pressure for X, Y dims
    // set nDimsBarostatted to 2
    pFlags[0] = true;
    pFlags[1] = true;
    pFlags[2] = true;

    // set Z flag to false if we are a 2d system
    if ( (state->is2d)) {
        pFlags[2] = false;
    }
    requiresVirials = true;
}


// pressure can be constant double value
void FixNoseHoover::setPressure(std::string pressMode, double press, double timeConstant) {
    // get the pressmode and couplestyle; parseKeyword also changes the 
    // state of the pressComputer & tempComputer if needed; alters the boolean flags for 
    // the dimensions that will be barostatted (XYZ, or XY).
    parseKeyword(pressMode);
    pressInterpolator = Interpolator(press);
    barostatting = true;
    pFrequency = 1.0 / timeConstant;
}

// could also be a python function
void FixNoseHoover::setPressure(std::string pressMode, py::object pressFunc, double timeConstant) {
    parseKeyword(pressMode);
    pressInterpolator = Interpolator(pressFunc);
    barostatting = true;
    pFrequency = 1.0 / timeConstant;
}

// could also be a list of set points with accompanying intervals (denoted by turns - integer values)
void FixNoseHoover::setPressure(std::string pressMode, py::list pressures, py::list intervals, double timeConstant) {
    parseKeyword(pressMode);
    pressInterpolator = Interpolator(pressures, intervals);
    barostatting = true;
    pFrequency = 1.0 / timeConstant; 
}


// and analogous procedure with setting the temperature
void FixNoseHoover::setTemperature(double temperature, double timeConstant) {
    tempInterpolator = Interpolator(temperature);
    frequency = 1.0 / timeConstant;

}

void FixNoseHoover::setTemperature(py::object tempFunc, double timeConstant) {
    tempInterpolator = Interpolator(tempFunc);
    frequency = 1.0 / timeConstant;
}

void FixNoseHoover::setTemperature(py::list intervals, py::list temps, double timeConstant) {
    tempInterpolator = Interpolator(intervals, temps);
    frequency = 1.0 / timeConstant;
}



//bool FixNoseHoover::prepareForRun()
bool FixNoseHoover::prepareFinal()
{

    // if we are barostatting, we need the virials.
    // if this is the first time that prepareForRun was called, we do not have them
    // so, return false and it'll get called again
    /*
    if (firstPrepareCalled && barostatting) {
        firstPrepareCalled = false;
        return false;

    }
    */
    // get our boltzmann constant
    boltz = state->units.boltz;

    // get number of atoms in the system
    nAtoms = state->atoms.size();
    // Calculate current kinetic energy
    tempInterpolator.turnBeginRun = state->runInit;
    tempInterpolator.turnFinishRun = state->runInit + state->runningFor;

    tempComputer.prepareForRun();
    
    calculateKineticEnergy();
    
    tempInterpolator.computeCurrentVal(state->runInit);
    setPointTemperature = tempInterpolator.getCurrentVal();
    oldSetPointTemperature = setPointTemperature;

    updateThermalMasses();

    // Update thermostat forces
    double temp = tempInterpolator.getCurrentVal();
    thermForce.at(0) = (ke_current - ndf * boltz * temp) / thermMass.at(0);
    for (size_t k = 1; k < chainLength; ++k) {
        thermForce.at(k) = (
                thermMass.at(k-1) *
                thermVel.at(k-1) *
                thermVel.at(k-1) - boltz*temp
            ) / thermMass.at(k);
    }

    // we now have the temperature set point value and instantaneous value.
    // -- set up the pressure set point value via pressInterpolator, 
    //    and get the instantaneous pressure.
    //   -- set initial values for assorted barostat and barostat-thermostat mass parameters
    //      analogous to what was done above
    if (barostatting) {
        // pressComputer, tempComputer were already set to appropriate "scalar"/"tensor" values
        // in the parseKeyword() call in the pertinent setPressure() function
        
        // get the number of dimensions barostatted
        nDimsBarostatted = 0;
        // go up to 6 - eventually we'll want Rahman-Parinello stress ensemble
        // -- for now, the max value of nDimsBarostatted is 3.
        for (int i = 0; i < 6; i++) {
            if (pFlags[i]) {
                nDimsBarostatted += 1;
            }
        }

        // set up our pressure interpolator
        pressInterpolator.turnBeginRun = state->runInit;
        pressInterpolator.turnFinishRun = state->runInit + state->runningFor;
        pressInterpolator.computeCurrentVal(state->runInit);
        
        setPointPressure = pressInterpolator.getCurrentVal();
        oldSetPointPressure = setPointPressure;
        // call prepareForRun on our pressComputer
        pressComputer.prepareForRun();

        // our P_{ext}, the set point pressure; this will need to change when we have 
        // the iso-stress ensemble; but, it suffices for now.
        hydrostaticPressure = pressInterpolator.getCurrentVal();

        // using external temperature... so send tempNDF and temperature scalar/tensor 
        // to the pressComputer, then call [computeScalar/computeTensor]_GPU() 
        if (pressMode == PRESSMODE::ISO) {
            double scaledTemp = currentTempScalar;
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = scaledTemp;
            pressComputer.computeScalar_GPU(true, groupTag);
        } else if (pressMode == PRESSMODE::ANISO) {

            Virial tempTensor_current = tempComputer.tempTensor;

            Virial scaledTemp = Virial(tempTensor_current[0], 
                                       tempTensor_current[1],
                                       tempTensor_current[2],
                                       0, 0, 0);

            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = scaledTemp;
            pressComputer.computeTensor_GPU(true, groupTag);

        } 

        // synchronize devices after computing the pressure..
        cudaDeviceSynchronize();

        // from GPU data, tell pressComputer to compute pressure on CPU side
        // --- might consider a boolean template argument for inside run() functions
        //     whether or not it is pressmode iso or aniso..
        //     --- once we go beyond iso&aniso we might make it class template
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.computeScalar_CPU();
        } else {
            pressComputer.computeTensor_CPU();
        }

        // get the instantaneous pressure from pressComputer; store it locally as a tensor
        // --- some redundancy here if we are using a scalar
        getCurrentPressure();

        // initialize the pressMass, pressVel, pressForce
        // -- one for each: xx yy zz xy xz yz
        //    although we really only care about the normal components in this implementation
        pressMass = std::vector<double> (6, 0.0);
        pressVel  = std::vector<double> (6, 0.0);
        pressForce= std::vector<double> (6, 0.0);

        // and barostat thermostat variables: pressThermMass, pressThermVel, and pressThermForce, respectively
        pressThermMass = std::vector<double> (pchainLength, 0.0);
        pressThermVel  = std::vector<double> (pchainLength, 0.0);
        pressThermForce= std::vector<double> (pchainLength, 0.0);

        // sanity check, set the above to zero
        for (int i = 0; i<pressMass.size(); i++) {
            pressMass[i]  = 0.0;
            pressVel[i]   = 0.0;
            pressForce[i] = 0.0;
        }

        // and same for the barostat thermostats
        for (int i = 0; i<pressThermMass.size(); i++) {
            pressThermMass[i] = 0.0;
            pressThermVel[i]  = 0.0;
            pressThermForce[i]= 0.0;
        }

        // -- set to the current set point
        //    we computed the set point above prior to calling updateThermalMasses()
        setPointTemperature = tempInterpolator.getCurrentVal();
        oldSetPointTemperature = setPointTemperature;
        updateBarostatMasses(false);
        updateBarostatThermalMasses(false);
        
    }

    prepared = true;
    return prepared;
}
bool FixNoseHoover::postRun()
{
    tempInterpolator.finishRun();
    rescale();

    return true;
}

bool FixNoseHoover::stepInit()
{

    // see Martyna et. al. 2006 for clarification of notation, p. 5641
    // lists the complete propagator used here.

    // -- step init: update set points and associated variables before we do anything else
    if (barostatting) {
        // update the set points for:
        // - pressures    (--> and barostat mass variables accordingly)
        // - temperatures (--> barostat thermostat's masses, particle thermostat's masses)

        // save old values before computing new ones
        oldSetPointPressure = setPointPressure;
        oldSetPointTemperature = setPointTemperature;

        // compute set point pressure, and save it to our local variable setPointPressure
        pressInterpolator.computeCurrentVal(state->turn);
        setPointPressure = pressInterpolator.getCurrentVal();

        // compute set point temperature, and save it to our local variable setPointTemperature
        tempInterpolator.computeCurrentVal(state->turn);
        setPointTemperature = tempInterpolator.getCurrentVal();
        
        // compare values and update accordingly
        // update the masses associated with thermostats for the barostats and the particles
        updateBarostatMasses(true);
        updateBarostatThermalMasses(true);
        updateThermalMasses();
        // exp(iL_{T_{BARO} \frac{\Delta t}{2})
        // -- variables that must be initialized/up-to-date:
        barostatThermostatIntegrate(true);

        // exp(iL_{T_{PART}} \frac{\Delta t}{2})
        // - does thermostat scaling of particle velocities
        thermostatIntegrate(true);

        // apply the thermostat scaling to the velocities
        rescale();

        // compute the kinetic energy of the particles
        // --- TODO: simple optimization here - we can save the scale factor and do the rescaling later
        //           - just need to make sure that we send the correct external temperature to pressComputer immediately below
        calculateKineticEnergy();

        // a brief diversion from the propagators: we need to tell the GPU to do the summations of
        // of the virials to get the current pressure tensor
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = currentTempScalar;
            pressComputer.computeScalar_GPU(true, groupTag);
            cudaDeviceSynchronize();
            pressComputer.computeScalar_CPU();
        } else if (pressMode == PRESSMODE::ANISO) {
            Virial tempTensor_current = tempComputer.tempTensor;
            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = tempTensor_current;
            pressComputer.computeTensor_GPU(true,groupTag);
            cudaDeviceSynchronize();
            pressComputer.computeTensor_CPU();
        }
        
        // and the current hydrostatic pressure (we computed this above already)
        hydrostaticPressure = setPointPressure;
        
        // and get the current pressure to our local variables; here we do the partitioning according to 
        // the couple style: {NONE,XYZ}
        getCurrentPressure();

        // exp(iL_{\epsilon_2} \frac{\Delta t}{2})
        // -- barostat velocities from virial, including the $\alpha$ factor 1+ 1/N_f
        // -- note that we modified the kinetic energy above via the thermostat
        //    - but, we also called calculateKineticEnergy() which updated this. so, nothing to do here right now
        barostatVelocityIntegrate();

        // after this, we exit stepInit, because we need IntegratorVerlet() to do a velocity timestep
        // --- THEN, we do barostat rescaling of velocities
        return true;

    } else {
        
        oldSetPointTemperature = setPointTemperature;
        tempInterpolator.computeCurrentVal(state->turn);
        setPointTemperature = tempInterpolator.getCurrentVal();
        // compare values and update accordingly
        
        if (oldSetPointTemperature != setPointTemperature) {
            // update the masses associated with thermostats for the particles
            updateThermalMasses();
        }

        thermostatIntegrate(true);
        
        rescale();
        

        return true;
        
    }

}

bool FixNoseHoover::postNVE_V() {
   
    if (barostatting) {
        // exp(iL_2 \frac{\Delta t}{2}
        // -- we have done our velocity integration in the integrator
        //    we now do the additive scaling of the forces (sinh(x)/x)
        //    followed by a multiplicative scaling of the resulting velocities

        // scale particle velocities due to barostat variables
        scaleVelocitiesBarostat(true);

        // and our operator acting on epsilon (volume rescale) --> changes particle positions as well
        rescaleVolume();
    }
    return true;

}

bool FixNoseHoover::postNVE_X() {
    // nothing to do here for these equations of motion.  We return immediately to integration of the 
    // velocities.
    return true;
}

bool FixNoseHoover::stepFinal()
{
    // at this point we have performed our second velocity verlet update of the particle velocities

    // - do barostat scaling of velocities
    if (barostatting) {
        
        // exp(iL_2 \frac{\Delta t}{2}) -- barostat rescaling of velocities component
        scaleVelocitiesBarostat(false);

        // update the kinetic energy, and compute the internal pressure
        calculateKineticEnergy();
        
        if (pressMode == PRESSMODE::ISO) {
            pressComputer.tempNDF = ndf;
            pressComputer.tempScalar = currentTempScalar;
            pressComputer.computeScalar_GPU(true, groupTag);
            cudaDeviceSynchronize();
            pressComputer.computeScalar_CPU();
        } else if (pressMode == PRESSMODE::ANISO) {
            Virial tempTensor_current = tempComputer.tempTensor;
            pressComputer.tempNDF = ndf;
            pressComputer.tempTensor = tempTensor_current;
            pressComputer.computeTensor_GPU(true,groupTag);
            cudaDeviceSynchronize();
            pressComputer.computeTensor_CPU();
        }

        // and get the current pressure to our local variables; here we do the partitioning according to 
        // the couple style: {NONE,XYZ}
        getCurrentPressure();

        // and the current hydrostatic pressure
        hydrostaticPressure = setPointPressure;

        // exp(iL_{\epsilon_2} \frac{\Delta t}{2})
        // integration of barostat velocities
        barostatVelocityIntegrate();

        // exp(iL_{T_{PART}} \frac{\Delta t}{2})
        // scaling of particle velocities from particle thermostats
        thermostatIntegrate(false);

        // exp(iL_{T_{BARO}} \frac{\Delta t}{2})
        // scaling of barostat velocities from barostat thermostats
        updateBarostatMasses(false);
        updateBarostatThermalMasses(false);
        barostatThermostatIntegrate(false);

    } else {
        // just thermostatting
        calculateKineticEnergy();

        thermostatIntegrate(false);

    }
     

    return true;

}

// save instantaneous pressure locally, and partition according to COUPLESTYLE::{XYZ,NONE}
void FixNoseHoover::getCurrentPressure() {

    // based off of the virial class
    if (pressMode == PRESSMODE::ISO) {
        double pressureScalar = pressComputer.pressureScalar;
        currentPressure = Virial(pressureScalar, pressureScalar, pressureScalar, 0, 0, 0);
    } else {
        Virial pressureTensor = pressComputer.pressureTensor;
        // partition the pressure;
        if (couple == COUPLESTYLE::XYZ) {
            // the virial pressure tensor in pressComputer goes as [xx,yy,zz,xy,xz,yz] 
            // (same as /src/Virial.h)
            double pressureScalar = (1.0 / 3.0) * (pressureTensor[0] + pressureTensor[1] + pressureTensor[2]);
            currentPressure = Virial(pressureScalar, pressureScalar, pressureScalar, 0, 0, 0);
        } else {
            currentPressure = pressureTensor;
        }
    }
}

// update barostat masses to reflect a change in the set point pressure
void FixNoseHoover::updateBarostatMasses(bool stepInit) {

    // set point temperature is of class Virial
    double t_external = setPointTemperature;
    if (stepInit) {
        // if we are at the initial step, use the old set point
        // -- this is due to ordering of the louiviliian propagators
        t_external = oldSetPointTemperature;
    }
    
    // the barostat mass expression is given in MTK 1994: Constant Pressure molecular dynamics algorithms
    // (1) isotropic: W = (N_f + d) kT / \omega_b^2
    // (2) anisotropic: W_g = W_g_0 = (N_f + d) kT / (d \omega_b^2)

    // 'N_f' number of degrees of freedom
    // -- held in our class variable ndf, from the tempComputer.ndf value
    //    (see ::calculateKineticEnergy())

    // 'd' - dimensionality of the system
    double d = 3.0;

    if (state->is2d) {
        d = 2.0;
    }

    double kt = boltz * t_external;
    // from MTK 1994
    // this has to be turned in to a \sum 
    if (pressMode == PRESSMODE::ISO) {
        for (int i = 0; i < 3; i++) {
        // then we set the masses to case (1)
            pressMass[i] = (ndf + d) * kt / (pFrequency * pFrequency);
        }
    } else {
        for (int i = 0; i < 3; i++) {
            pressMass[i] = (ndf + d) * kt / (d * pFrequency * pFrequency);
        }
    }
}

void FixNoseHoover::updateBarostatThermalMasses(bool stepInit) {

    // from MTK 1994:
    // Q_b_1 = d(d+1)kT/(2 \omega_b^2)
    double t_external = setPointTemperature;
    if (stepInit) {
        t_external = oldSetPointTemperature;
    }

    double kt = boltz * t_external;

    double d = 3.0;
    if (state->is2d) {
        d = 2.0;
    }

    pressThermMass[0] = d*(d+1.0) * kt / (2.0 * pFrequency);

    // Q_b_i = kt/(\omega_i^2)
    for (int i = 1; i < pchainLength; i++) {
        pressThermMass[i] = kt / (pFrequency * pFrequency);
    }

    // the forces are functions of the thermal masses; update these as well
    // -- pressThermForce[0] is not acted upon here
    for (int i = 1; i < pchainLength; i++) {
        pressThermForce[i] = ((pressThermMass[i-1] * pressThermVel[i-1] * 
                               pressThermVel[i-1] - kt) / (pressThermMass[i]));
    }


}

void FixNoseHoover::barostatThermostatIntegrate(bool stepInit) {

    // as thermostatIntegrate, get the set point temperature
    double kt = boltz * setPointTemperature;

    if (stepInit) {
        kt = boltz * oldSetPointTemperature;
    }

    // calculate the kinetic energy of our barostats - 
    //   only the dimensions we are barostatting.
    //   e.g., if 2D, we don't count Z (although it should be zero anyways)
    double ke_barostats = 0.0;
    for (int i = 0; i < 6; i++) {
        if (pFlags[i]) {
            ke_barostats += (pressMass[i] * pressVel[i] * pressVel[i]);
        }
    }

    if (!stepInit) {
        pressThermForce[0] = (ke_barostats - kt) / (pressThermMass[0]);
    }

    
    // this is the same routine as in thermostatIntegrate
    for (size_t i = 0; i < nTimesteps_b; ++i) {
        for (size_t j = 0; j < n_ys_b; ++j) {
            double timestep = weight.at(j)*state->dt / nTimesteps_b;
            double timestep2 = 0.5*timestep;
            double timestep4 = 0.25*timestep;
            double timestep8 = 0.125*timestep;

            // Update thermostat velocities
            pressThermVel.back() += timestep4*pressThermForce.back();
            for (size_t k = pchainLength-2; k > 0; --k) {
                double preFactor = std::exp( -timestep8*pressThermVel.at(k+1) );
                pressThermVel.at(k) *= preFactor;
                pressThermVel.at(k) += timestep4 * pressThermForce.at(k);
                pressThermVel.at(k) *= preFactor;
            }

            double preFactor = std::exp( -timestep8*pressThermVel.at(1) );
            pressThermVel.at(0) *= preFactor;
            pressThermVel.at(0) += timestep4*pressThermForce.at(0);
            pressThermVel.at(0) *= preFactor;

            // Update particle (barostat) velocities
            double barostatScaleFactor = std::exp( -timestep2*pressThermVel.at(0) );

            // apply the scaling of the barostat velocities
            for (int i = 0; i < 6; i++) {
                if (pFlags[i]) {
                    pressVel[i] *= barostatScaleFactor;
                }
            }

            // as done in particle thermostatting, get new ke_current (ke_barostats)
            ke_barostats = 0.0;
            for (int i = 0; i < 6; i++) {
                if (pFlags[i]) {
                    ke_barostats += (pressMass[i] * pressVel[i] * pressVel[i]);
                }
            }

            // Update the forces
            pressThermVel.at(0) *= preFactor;
            pressThermForce.at(0) = (ke_barostats - kt) / pressThermMass.at(0);
            pressThermVel.at(0) += timestep4 * pressThermForce.at(0);
            pressThermVel.at(0) *= preFactor;

            // Update thermostat velocities
            for (size_t k = 1; k < pchainLength-1; ++k) {
                preFactor = std::exp( -timestep8*pressThermVel.at(k+1) );
                pressThermVel.at(k) *= preFactor;
                pressThermForce.at(k) = (
                        pressThermMass.at(k-1) *
                        pressThermVel.at(k-1) *
                        pressThermVel.at(k-1) - kt
                    ) / pressThermMass.at(k);
                pressThermVel.at(k) += timestep4 * pressThermForce.at(k);
                pressThermVel.at(k) *= preFactor;
            }

            pressThermForce.at(pchainLength-1) = (
                    pressThermMass.at(pchainLength-2) *
                    pressThermVel.at(pchainLength-2) *
                    pressThermVel.at(pchainLength-2) - kt
                ) / pressThermMass.at(pchainLength-1);
            pressThermVel.at(pchainLength-1) += timestep4*pressThermForce.at(pchainLength-1);
        }
    }
}

void FixNoseHoover::thermostatIntegrate(bool stepInit) {
 // Equipartition at desired temperature
    // setPointTemperature should be up to date.
    double nkt = ndf * boltz * setPointTemperature;

    double temp = setPointTemperature;
    if (!stepInit) {
        thermForce.at(0) = (ke_current - nkt) / thermMass.at(0);
    }

    // Multiple timestep procedure
    for (size_t i = 0; i < nTimesteps; ++i) {
        for (size_t j = 0; j < n_ys; ++j) {
            double timestep = weight.at(j)*state->dt / nTimesteps;
            double timestep2 = 0.5*timestep;
            double timestep4 = 0.25*timestep;
            double timestep8 = 0.125*timestep;

            // Update thermostat velocities
            thermVel.back() += timestep4*thermForce.back();
            for (size_t k = chainLength-2; k > 0; --k) {
                double preFactor = std::exp( -timestep8*thermVel.at(k+1) );
                thermVel.at(k) *= preFactor;
                thermVel.at(k) += timestep4 * thermForce.at(k);
                thermVel.at(k) *= preFactor;
            }

            double preFactor = std::exp( -timestep8*thermVel.at(1) );
            thermVel.at(0) *= preFactor;
            thermVel.at(0) += timestep4*thermForce.at(0);
            thermVel.at(0) *= preFactor;

            // Update particle velocities
            double scaleFactor = std::exp( -timestep2*thermVel.at(0) );
            scale *= scaleFactor;

            ke_current *= scaleFactor*scaleFactor;

            // Update the thermostat positions
            //for (size_t k = 0; k < chainLength; ++k) {
            //    thermPos.at(k) += timestep2*thermVel.at(k);
            //}

            // Update the forces
            thermVel.at(0) *= preFactor;
            thermForce.at(0) = (ke_current - nkt) / thermMass.at(0);
            thermVel.at(0) += timestep4 * thermForce.at(0);
            thermVel.at(0) *= preFactor;

            // Update thermostat velocities
            for (size_t k = 1; k < chainLength-1; ++k) {
                preFactor = std::exp( -timestep8*thermVel.at(k+1) );
                thermVel.at(k) *= preFactor;
                thermForce.at(k) = (
                        thermMass.at(k-1) *
                        thermVel.at(k-1) *
                        thermVel.at(k-1) - boltz*temp
                    ) / thermMass.at(k);
                thermVel.at(k) += timestep4 * thermForce.at(k);
                thermVel.at(k) *= preFactor;
            }

            thermForce.at(chainLength-1) = (
                    thermMass.at(chainLength-2) *
                    thermVel.at(chainLength-2) *
                    thermVel.at(chainLength-2) - boltz*temp
                ) / thermMass.at(chainLength-1);
            thermVel.at(chainLength-1) += timestep4*thermForce.at(chainLength-1);
        }
    }

}

void FixNoseHoover::barostatVelocityIntegrate() {

    // $G_{\epsilon}$ = \alpha * (ke_current) + (Virial - P_{ext})*V

    // so, we need to have the /current pressure/
    //  we need to have the /current kinetic energy of the particles/
    //  we need to have the instantaneous volume
    //  also, note that the deformations of slant vectors are not affected 
    //  by the external pressure P_{ext}, should we incorporate this later

    // instantaneous pressure
    Virial P_inst = currentPressure;

    // external pressure (the set point)
    double P_ext = setPointPressure;
    
    // current volume of our simulation cell
    double volume = state->boundsGPU.volume();
    
    // our sum over the particles: 
    std::vector<double> aggregate = std::vector<double> (6, 0.0);

    // for dN factor in denominator
    double d = 3.0;
    if (state->is2d) d = 2.0;
   
    
    Virial G_e = Virial(0, 0, 0, 0, 0, 0);
    Virial mvv = Virial(0, 0, 0, 0, 0, 0);

    double alphaAddition = 0.0;
    // get the extraneous alpha * mvv contribution; note that 
    // the temperature is either as a tensor or a scalar, so make considerations for that here
    // TODO: should nAtoms, ndf be considered only from number of atoms to which this fix is being applied?
    //       i.e. what is we are not thermostatting all particles in the simulation? needs further consideration
    if (pressMode == PRESSMODE::ISO) {
        // currentTempScalar in ::calculateKineticEnergy is the scalar temperature
        alphaAddition = boltz * currentTempScalar * ndf / (state->atoms.size() * d);
    } else {
        // this should be up-to-date at this point
        // --take the average as the alpha contribution
        Virial tempTensor = tempComputer.tempTensor;
        for (int i = 0; i < 3; i++) {
            // tempTensor incorporates tdof
            alphaAddition += tempTensor[i];
        }

        alphaAddition /= (d * state->atoms.size());
    }
                
    // get the pressure differential contribution
    // -- add the mvv*alpha contribution. unit conversion?
    // -- divide by pressMass, because we are evolving the velocity - not the momenta            
    for (int i = 0; i < 3; i++) {
        if (pFlags[i]) {
            G_e[i] = alphaAddition / pressMass[i];
            G_e[i] += ( (P_inst[i] - P_ext) * volume / (pressMass[i] * state->units.nktv_to_press));
        }
    }

    // for completeness, propagate the skew vectors as well
    for (int i = 3; i < 6; i++) {
        if (pFlags[i]) {
            G_e[i] += ( (P_inst[i]  * volume) / (pressMass[i] * state->units.nktv_to_press));
        }
    }

    // and evolve the velocity of the barostat variables according to G_e[i]
    // here we recognize the action of the following operator:
    // exp(iL_{\epsilon_2} \frac{\Delta t}{2}) in MTK 2006
    for (int i = 0; i < 6; i++) {
        pressVel[i] += (G_e[i] * state->dt * 0.5) ;
    }


}

void FixNoseHoover::scaleVelocitiesBarostat(bool preNVE_X) {
    // bool input denotes whether we are pre or post integration of positions in the 
    // velocity-verlet step
    // --- does anything change pre- versus post- NVE_X ?
    float d = 3.0;
    if (state->is2d) d = 2.0;

    float alphaAddition = 0.0;
    for (int i = 0; i < 3; i++) {
        alphaAddition += pressVel[i];
    }
    alphaAddition /= (d * nAtoms);
    // at this point, velocities of the barostat variables are up-to-date
    float3 velScaleAdditive = make_float3(0.0, 0.0, 0.0);

    float timestep4 = state->dt * 0.25f;

    // compute the additive velocity scale... sinh(x) / x power series
    float3 v_eps = make_float3(pressVel[0] + alphaAddition, 
                               pressVel[1] + alphaAddition, 
                               pressVel[2] + alphaAddition);
    float3 x = 1.0 * v_eps * timestep4;

    // get our terms
    float3 x2 = x*x ;
    float3 x4 = x2*x2;
    float3 x6 = x4*x2;
    float3 x8 = x4*x4;

    // 6.0, 120.0, 5040.0, 362880.0, are 3!, 5!, 7!, and 9!, respectively, (!) being the factorial operator
    // -- this is a simple power series expansion of sinh(x)/x, where 
    // x is $v_{\epsilon}  \alpha  \frac{\Delta t}{4}$ as in MTK 2006
    x2 /= 6.0;
    x4 /= 120.0;
    x6 /= 5040.0;
    x8 /= 362880.0;

    // note that there is a +1.0 term leading the expansion.
    // but, we subtract 1.0 from the entire series, so we omit that here
    velScaleAdditive = x2 + x4 + x6 + x8;

    // and now do the velScaleMultiplicative
    // exp(-\alpha v_{\epsilon} dt / 2.0)
    // we write it here in terms of our variable 'x' define above
    // -- for the multiplicative velocity scaling, we have -dt/2 factor rather than dt/4
    //    so just multiply by -2.0
    float3 velScaleMultiplicative = make_float3(std::exp(-1.0*x.x),
                                                std::exp(-1.0*x.y),
                                                std::exp(-1.0*x.z));

    // and now send the atoms velocities and forces to a kernel and do the computations
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    if (groupTag == 1) {
        barostat_vel_no_tags_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(
                                                 nAtoms,
                                                 state->gpd.vs.getDevData(),
                                                 state->gpd.fs.getDevData(),
                                                 velScaleAdditive,
                                                 velScaleMultiplicative,
                                                 dtf);
    } else {
        barostat_vel_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms,
                                                 groupTag,
                                                 state->gpd.vs.getDevData(),
                                                 state->gpd.fs.getDevData(),
                                                 velScaleAdditive,
                                                 velScaleMultiplicative,
                                                 dtf);
    }

}

void FixNoseHoover::rescaleVolume() {

    float3 volScaleXYZ = make_float3(1.0, 1.0, 1.0);
    float dt = state->dt;

    // set the x, y scale factors; if this is a 3d simulation, pFlags[2] will evaluate to true
    volScaleXYZ.x = std::exp(pressVel[0] * dt);
    volScaleXYZ.y = std::exp(pressVel[1] * dt);

    if (pFlags[2]) {
        volScaleXYZ.z = std::exp(pressVel[2] * dt);
    }

    Mod::scaleSystem(state, volScaleXYZ, groupTag);

}

void FixNoseHoover::updateThermalMasses()
{
    double temp = tempInterpolator.getCurrentVal();
    thermMass.at(0) = ndf * boltz * temp / (frequency*frequency);
    for (size_t i = 1; i < chainLength; ++i) {
        thermMass.at(i) = boltz*temp / (frequency*frequency);
    }
}


void FixNoseHoover::calculateKineticEnergy()
{
    if (not barostatting) {
        tempComputer.computeScalar_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeScalar_CPU();
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;
    } else if (pressMode == PRESSMODE::ISO) {
        tempComputer.computeScalar_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeScalar_CPU();
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;

       // tempComputer.computeTensorFromScalar();
    } else if (pressMode == PRESSMODE::ANISO) {
        tempComputer.computeTensor_GPU(true, groupTag);
        cudaDeviceSynchronize();
        tempComputer.computeTensor_CPU();

        tempComputer.computeScalarFromTensor(); 
        ndf = tempComputer.ndf;
        ke_current = tempComputer.totalKEScalar;
        //need this for temp biz
    } 

    // set class variable currentTempScalar to value.
    // -- this way, it is always up-to-date (less scale factors, when those are implemented)
    currentTempScalar = tempComputer.tempScalar;
}

void FixNoseHoover::rescale()
{
    if (scale == make_float3(1.0f, 1.0f, 1.0f)) {
        return;
    }

    if (groupTag == 1) {
        rescale_no_tags_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(
                                                 nAtoms,
                                                 state->gpd.vs.getDevData(),
                                                 scale);
    } else {
        rescale_cu<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms,
                                                 groupTag,
                                                 state->gpd.vs.getDevData(),
                                                 state->gpd.fs.getDevData(),
                                                 scale);
    }

    scale = make_float3(1.0f, 1.0f, 1.0f);
}

Interpolator *FixNoseHoover::getInterpolator(std::string type) {
    if (type == "temp") {
        return &tempInterpolator;
    }
    return nullptr;
}

// setting up a few exports for BOOST
void (FixNoseHoover::*setTemperature_x2) (py::object, double) = &FixNoseHoover::setTemperature;
void (FixNoseHoover::*setTemperature_x1) (double, double) = &FixNoseHoover::setTemperature;
void (FixNoseHoover::*setTemperature_x3) (py::list, py::list, double) = &FixNoseHoover::setTemperature;

void (FixNoseHoover::*setPressure_x2) (std::string, py::object, double) = &FixNoseHoover::setPressure;
void (FixNoseHoover::*setPressure_x1) (std::string, double, double) = &FixNoseHoover::setPressure;
void (FixNoseHoover::*setPressure_x3) (std::string, py::list, py::list, double) = &FixNoseHoover::setPressure;


void export_FixNoseHoover()
{
    py::class_<FixNoseHoover,                    // Class
               boost::shared_ptr<FixNoseHoover>, // HeldType
               py::bases<Fix>,                   // Base class
               boost::noncopyable>
    (
        "FixNoseHoover",
        py::init<boost::shared_ptr<State>, std::string, std::string>(
            py::args("state", "handle", "groupHandle")
        )
    )
    .def("setTemperature", setTemperature_x2,
         (py::arg("tempFunc"),
          py::arg("timeConstant")
         )
        )
    .def("setTemperature", setTemperature_x1,
         (py::arg("temp"),
          py::arg("timeConstant")
         )
        )
    .def("setTemperature", setTemperature_x3,
         (py::arg("intervals"),
          py::arg("tempList"),
          py::arg("timeConstant")
         )
        )
    .def("setPressure", setPressure_x2,
         (py::arg("mode"), 
          py::arg("pressureFunc"),
          py::arg("timeConstant")
         )
        )
    .def("setPressure", setPressure_x1,
         (py::arg("mode"), 
          py::arg("pressure"),
          py::arg("timeConstant")
         )
        )
    .def("setPressure", setPressure_x3,
         (py::arg("mode"), 
          py::arg("pressureList"),
          py::arg("intervals"),
          py::arg("timeConstant")
         )
        )
    ;

}
