#pragma once
#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python/list.hpp>
#include <string>
#include <vector>
#include "IntegratorUtil.h"

class GPUArray;
class State;

void export_Integrator();

//! Base class for Molecular Dynamics Integrators
/*
 * This class is a base class for all MD Integrators. It takes care of all
 * aspects common to all integrators such as doing basic checks, data transfer
 * from and to the GPU, etc.
 */
class Integrator : public IntegratorUtil {

protected:
    //! Call fixes just before a step
    void stepInit(bool computeVirials);


    
    //! Call fixes just after the timestepping
    void stepFinal();

    //! Perform all asynchronous operations
    /*!
     * This function performs all asynchronous operations, such as writing
     * configurations or performing Python operations
     */
    void asyncOperations();
    std::vector<GPUArray *> activeData; //!< List of pointers to the data
                                        //!< used by this integrator

    //! Simple checks before the run
    /*!
     * The checks consist of:
     *   - GPU device compatibility needs to be >= 3.0
     *   - Atom grid needs to be set
     *   - Cutoff distance needs to be set
     *   - 2d system may not be periodic in z dimension
     *   - Grid discretization must not be smaller than cutoff distance plus
     *     padding
     */
    void basicPreRunChecks();

    //! Prepare Integrator for running
    /*!
     * \param numTurns Number of turns the integrator is expected to run
     *
     * Prepare the integrator to run for a given amount of timesteps. This
     * includes copying all data to the GPU device and calling prepareForRun()
     * on all fixes.
     */
    std::vector<bool> basicPrepare(int numTurns);

    //! Finish simulation run
    /*!
     * Finish the simulation run. This includes copying all relevant data to
     * the CPU host and calling postRun on all fixes.
     */
    void basicFinish();

    //! Collect all pointers to the relevant data into activeData
    void setActiveData();

    //set runtime tunable parameters for performance
    double tune();




public:
    //! Calculate and return single point energy
    /*!
     * \param groupHandle Handle defining the group used for averaging
     *
     * \return Average energy for all particles in the specified group
     *
     * This function calculates and returns the average per particle energy for
     * the particles in the group specified via the groupHandle.
     */
    //SHOULD THIS BE HERE OR IN STATE?
    //double singlePointEngPythonAvg(std::string groupHandle);

    //! Create list of per-particle energies
    /*!
     * \return List containing the per-particle energy for each atom
     *
     * This function calculates the per-particle energy and returns a list
     * containing one value per atom in the simulation.
     */
    //boost::python::list singlePointEngPythonPerParticle();

    //! Default constructor
    /*!
     * \todo Do we need a default constructor?
     */
    Integrator() {};

    //! Constructor
    /*!
     * \param state_ Pointer to simulation state
     * \param type_ String specifying type of Integrator (unused)
     *
     * \todo Remove usage of type
     */
    explicit Integrator(State *state_);



    //! Write output for all \link WriteConfig WriteConfigs \endlink
    void writeOutput();
    float dtf;
};

#endif
