#pragma once
#ifndef INITIALIZE_H
#define INTIIALIZE_H

#include <math.h>
#include <vector>
#include <map>
#include <random>

#include "boost_for_export.h"

#include "Python.h"

#include "globalDefs.h"
#include "Bounds.h"

void export_InitializeAtoms();

class State;

//! Number of tries to place Atom before throwing an error
const unsigned int maxtries = 10000;

/*! \class InitializeAtomsPythonWrap
 * \brief Python Wrapper for Atom initialization
 *
 * A small Python wrapper for the functions defined in the namespace
 * InitializeAtoms.
 */
class InitializeAtomsPythonWrap { };

/*! \brief Functions to add atoms to the simulation
 *
 * This namespace contains functions for adding new atoms to the simulation,
 * either placing them on a grid or placing them randomly, and initializing
 * them with random velocities.
 */
namespace InitializeAtoms {
    /*! \brief Add new atoms placed on a grid
     *
     * \param state Simulation to which the atoms are added.
     * \param bounds Region into which the atoms are placed.
     * \param handle String defining the type of the newly placed atoms.
     * \param n Number of atoms to add.
     *
     * This function adds n new atoms to the simulations. The atoms are placed
     * on a grid in the region specified by bounds. The grid is chosen such
     * that it has the maximum spacing while still being able to accomodate all
     * atoms. The number of grid points n_p is the same in x-, y-, and
     * z-direction and chosen as the smallest integer for which n_p^3 > n.
     *
     * \todo This will horribly fail in 2-d Simulation or in general when the
     *       bounds are small in one direction.
     */
    void populateOnGrid(boost::shared_ptr<State> state, Bounds &bounds,
                        std::string handle, int n);

    /*! \brief Add new atoms at random positions
     *
     * \param state Simulation to which the atoms are added.
     * \param bounds Region into which the atoms are placed.
     * \param handle String defining the type of the newly placed atoms.
     * \param n Number of atoms to add.
     * \param distMin Minimum distance between two atoms.
     *
     * This function places n new atoms randomly into the region specified by
     * the given bounds. For each atom a candidate position is determined from
     * a uniform distribution and this candidate position is rejected if it is
     * closer than distMin from any other atom in the simulation. Otherwise,
     * the candidate position is accepted and the atom added to the simulation.
     *
     * \todo If n is too large, the bounds too small or already crowded with
     *       atoms, this function will produce an endless loop. I suggest to
     *       use something like MAX_TRIES to abort when an atom can not be
     *       placed.
     */
    void populateRand(boost::shared_ptr<State> state, Bounds &bounds,
                      std::string handle, int n, double distMin);

    /*! \brief Give group of atoms random velocities
     *
     * \param state Simulation state.
     * \param groupHandle Handle specifying the group of atoms.
     * \param temp Temperature.
     *
     * This function attributes to each atom in the specified group a new
     * velocity. The velocity is generated from a normal distribution with
     * variance temp/mass, where the mass can be different for each atom.
     * Furthermore, the velocities are chosen such that the center of mass
     * motion is zero.
     */
    void initTemp(boost::shared_ptr<State> state,
                  std::string groupHandle, double temp);
}

#endif
