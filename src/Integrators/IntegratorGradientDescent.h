#pragma once

#include "Integrator.h"

//! Make the Integrator accessible to the Python interface
void export_IntegratorGradientDescent();

//! Velocity-Verlet integrator
/*!
 * This class implements the reversible version of the velocity-Verlet
 * integration scheme as described by Tuckerman et al.
 * \cite TuckermanEtal:JCP1992 .
 */
class IntegratorGradientDescent : public Integrator
{
public:
    //! Constructor
    /*!
     * \param statePtr Pointer to the simulation state
     */
    IntegratorGradientDescent(State *statePtr);

    //! Run the Integrator
    /*!
     * \param numTurns Number of steps to run
     */
    virtual void run(int numTurns, double coef);

private:
    
    /*!
     * Step along the gradient
     * 
     * 
     * 
     */
    void step(double coef);
    double getSumForceSqr();
};


