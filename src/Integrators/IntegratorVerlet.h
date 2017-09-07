#pragma once
#ifndef INTEGRATORVERLET_H
#define INTEGRATORVERLET_H

#include "Integrator.h"
class Interpolator;

//! Make the Integrator accessible to the Python interface
void export_IntegratorVerlet();

//! Velocity-Verlet integrator
/*!
 * This class implements the reversible version of the velocity-Verlet
 * integration scheme as described by Tuckerman et al.
 * \cite TuckermanEtal:JCP1992 .
 */
class IntegratorVerlet : public Integrator
{
public:
    //! Constructor
    /*!
     * \param statePtr Pointer to the simulation state
     */
    IntegratorVerlet(State *statePtr);

    //! Run the Integrator
    /*!
     * \param numTurns Number of steps to run
     */
    virtual double run(int numTurns);

private:
    //! Run first half-integration
    /*!
     * The first half-integration of the reversible velocity-Verlet scheme
     * integrates the velocities by half a timestep and the positions by a
     * full timestep.
     */
    void preForce();
    
    // BPK
    //void printConfig();
    // BPK END
    //
    //! Run second half-integration step
    /*!
     * \param index Active index of GPUArrayPairs
     *
     * The second half-integration of the reversible velocity-Verlet scheme
     * integrates the velocities by another half timestep such that velocities
     * and positions are in sync again.
     */
    void postForce();
    void nve_v();
    void nve_x();
    Interpolator *tempInterpolator; //for PIMD
    void setInterpolator();
};

#endif
