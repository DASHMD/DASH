#pragma once
#ifndef STATE_H
#define STATE_H

#define RCUT_INIT -1
#define PADDING_INIT 0.5

#include <assert.h>
#include <stdint.h>
#include <iostream>

#include <map>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <set>
#include <functional>
#include <random>
#include <thread>

#include <boost/shared_ptr.hpp>
#include <boost/type_traits/remove_cv.hpp> //boost 1.58 bug workaround
#include <boost/variant/get.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>

#include "globalDefs.h"
#include "GPUArrayTex.h"
#include "GPUArrayGlobal.h"
#include "GPUArrayDeviceGlobal.h"

#include "AtomParams.h"
#include "Atom.h"
#include "Molecule.h"
#include "Bond.h"
#include "GPUData.h"
#include "GridGPU.h"
#include "Bounds.h"
#include "DataManager.h"
#include "Group.h"

#include "boost_for_export.h"
#include "DeviceManager.h"
//basic integrator functions state may need access to (computing engs, for examples)
#include "IntegratorUtil.h"
#include "Units.h"


void export_State();

class PythonOperation;
class ReadConfig;
class Fix;
class WriteConfig;

enum EXCLUSIONMODE {FORCER, DISTANCE};
//! Simulation state
/*!
 * This class reflects the current state of the simulation. It contains and
 * manages all data.
 */
class State {
private:
    //! Remove a group tag from the list of group tags
    /*!
     * \param handle String specifying the group
     * \return Always true
     *
     * This function removes a group tag from the list of group tags.
     */
    bool removeGroupTag(std::string handle);

    //! Add a new group tag
    /*!
     * \param handle New group tag to be added
     * \return New group bitmask
     *
     * Add a new group tag and determine a new group tag bitmask. The bitmask
     * will be the smallest unsigned int that is not yet used by another group
     * tag.
     */
    uint addGroupTag(std::string handle);

    //! Get the maximum cutoff value from all fixes
    /*!
     * \return Maximum cutoff value
     *
     * To be called from within prepare for run __after__ fixed have prepared
     * (because that removes any DEFAULT_FILL values)
     */
    float getMaxRCut();

public:
    std::vector<Atom> atoms; //!< List of all atoms in the simulation
    boost::python::list molecules; //!< List of all molecules in the simulation.  Molecules are just groups of atom ids with some tools for managing them.  Using python list because users should to be able to 'hold on' to molecules without worrying about segfaults
    GridGPU gridGPU; //!< The Grid on the GPU
    BoundsGPU boundsGPU; //!< Bounds on the GPU
    GPUData gpd; //!< All GPU data
    DeviceManager devManager; //!< GPU device manager

    IntegratorUtil integUtil;
    Bounds bounds; //!< Bounds on the CPU
    std::vector<Fix *> fixes; //!< List of all fixes
    std::vector<boost::shared_ptr<Fix> > fixesShr; //!< List of shared pointers to fixes
    MD_ENGINE::DataManager dataManager; //!< Data manager
    std::vector<boost::shared_ptr<WriteConfig> > writeConfigs; //!< List of output writers
    std::vector<boost::shared_ptr<PythonOperation> > pythonOperations; //!< List of Python
                                                            //!< operations
    std::map<std::string, uint32_t> groupTags; //!< Map of group handles and
                                               //!< bitmasks
    std::map<uint32_t,Group> groups; //!< Map of group handles to a given group
    void populateGroupMap(); //!< Populates the data of our Group instances contained in the above map 'groups'

    bool is2d; //!< True for 2d simulations, else False
    bool periodic[3]; //!< If True, simulation is periodic in given dimension
    float dt; //!< Timestep
    float specialNeighborCoefs[3]; //!< Coefficients for modified neighbor
                                   //!< interactions of bonded atoms (1-2, 1-3,
                                   //!< 1-4 neighbors)
    int64_t turn; //!< Step of the simulation
    int runningFor; //!< How long the simulation is currently running
    int nlistBuildCount; //!< number of times we have build nlists
    std::vector<int> nlistBuildTurns; //!< turns at which we built the neighborlist
    int64_t runInit; //!< Timestep at which the current run started
    int64_t nextForceBuild; //!< Timestep neighborlists will definitely be build.  Fixes might need to request this
    int dangerousRebuilds; //!< Unused
    int periodicInterval; //!< Periodicity to wrap atoms and rebuild neighbor
                          //!< list
    bool requiresCharges; //!< Charges will be stored 
    bool requiresPostNVE_V;//!< If any of the need a step between post nve_v and nve_x.  If not, combine steps and do not call it.  If so, call it for all fixes

    //! Cutoff parameter for pair interactions
    /*!
     * Each pair fix can define its own cutoff distance. If no fix defines a
     * cutoff distance, this value is used as the default. When a run begins,
     * the grid will determine the maximum cutoff distance of all Fixes and use
     * that value. This bit has not been implemented yet.
     */
    double rCut;
    double padding; //!< Added to rCut for cutoff distance of neighbor building
    int exclusionMode; //!< Mode for handling bond list exclusions.  See comments for exclusions in GridGPU
    void setExclusionMode(std::string);

    // Variables that enable extension to PIMD
    int nPerRingPoly;			// RP discretization/number of time slices
					// possibly later allow this to be vector per atom 



    //! Set the coefficients for bonded neighbor interactions
    /*!
     * \param onetwo Parameter for 1-2 neighbors
     * \param onethree Parameter for 1-3 neighbors
     * \param onefour Parameter for 1-4 neighbors
     *
     * Interaction potentials such as FENE bonds or harmonic bonds require that
     * the interaction potentials between bonded atoms are modified. The
     * interaction is multiplied by a specified factor for 1-2, 1-3, and 1-4
     * neighbors. 1-2 neighbors are directly bonded, 1-3 neighbors are bonded
     * with one intermediate atom, and 1-4 neighbors are bonded with two
     * intermediate atoms.
     */
    void setSpecialNeighborCoefs(float onetwo, float onethree, float onefour);

    //! Add a Fix to the simulation
    /*!
     * \param other Fix to be added to the simulation state
     * \return True if the Fix was successfully added
     *
     * Add a new Fix to the Simulation. Note that the Fix has to be initialized
     * with the same State that it is added to.
     */
    bool activateFix(boost::shared_ptr<Fix> other);

    //! Remove a Fix from the simulation
    /*!
     * \param other Fix to be removed
     * \return False if Fix was not activated, else return True
     */
    bool deactivateFix(boost::shared_ptr<Fix> other);

    //! Specify output for the simulation
    /*!
     * \param other Configuration Writer to be added to the simulation
     * \return True if added successfully, else False
     */
    bool activateWriteConfig(boost::shared_ptr<WriteConfig> other);

    //! Remove output from simulation
    /*!
     * \param other Configuration Writer to be removed from the simulation
     * \return False if Writer was not previously added to the simulation
     */
    bool deactivateWriteConfig(boost::shared_ptr<WriteConfig> other);

    //! Add a Python operation to the simulation
    /*!
     * \param other Python operation to be added
     * \return True if operation was successfully added, else return False
     */
    bool activatePythonOperation(boost::shared_ptr<PythonOperation> other);

    //! Remove Python operation from the simulation
    /*!
     * \param other Python operation to be added
     * \return False if operation is not in the list of Python operations
     */
    bool deactivatePythonOperation(boost::shared_ptr<PythonOperation> other);

    //bool fixIsActive(boost::shared_ptr<Fix>);


    //! Add Atoms to a Group
    /*!
     * \param handle String specifying the atom group
     * \param toAdd List of Atoms to add to the group
     * \return True always
     *
     * This function adds a list of Atoms to a specified group. Note that the
     * group needs to be created first. This function does not create a new
     * group if it does not exist yet.
     *
     * This function is exposed to Python and accessed in Python via
     * addToGroup().
     */
    bool addToGroupPy(std::string handle, boost::python::list toAdd);

    //! Add Atoms to a group
    /*!
     * \param handle String specifying the group
     * \param testF Function pointer taking an Atom pointer and returning bool
     * \return True always
     *
     * For each Atom pointer, this function calls a test function and adds the
     * Atom to the group if the test function returns true.
     *
     * For this function the group must not exist yet. Instead, it is created
     * by this function.
     *
     * Note that the Python interface function addToGroup() does *not* call
     * this function but State::addToGroupPy().
     *
     * \todo For addToGroupPy() the group must already exist, for addToGroup()
     *       the group must not exist yet. This is *very* confusing. I suggest
     *       adding a bool createIfMissing = true parameter to determine
     *       whether the group should be created or if an error should be
     *       issued.
     *
     * \todo I find it mildly confusing that the Python addToGroup() command
     *       does not call addToGroup(), but addToGroupPy(). Can we resolve
     *       this? Maybe using overloaded functions?
     */
    bool addToGroup(std::string handle, std::function<bool (Atom *)> testF);

    //! Get list of all atoms in a specific group
    /*!
     * \param handle String specifying the group
     * \return vector of pointers for all Atoms in the group
     */
    std::vector<Atom *> selectGroup(std::string handle);

    //! Remove group from the simulation
    /*!
     * \param handle String specifying the group
     * \return True always
     *
     * Remove a group from the simulation. The group must exist. The group
     * "all" may not be removed.
     */
    bool deleteGroup(std::string);

    int countNumInGroup(std::string);
    int countNumInGroup(uint32_t);

    //! Create a new atom group
    /*!
     * \param handle String specifying the group
     * \param atoms List of atoms to add to the group
     * \return False if group already exists.
     *
     * Add a list of atoms to a group that is newly created in the process. If
     * the group already exists, no group is created and the atoms are not
     * added to any group.
     */
    bool createGroup(std::string handle,
                     boost::python::list atoms = boost::python::list());

    //! Get bitmask for a specific group
    /*!
     * \param handle String specifying the group
     * \return Bitmask, encoded into 32bit integer
     */
    uint32_t groupTagFromHandle(std::string handle);

    //! Add an Atom to the simulation
    /*!
     * \param handle String specifying the Atom type
     * \param pos Vector specifying the Atom position
     * \param q Charge of the atom
     * \return Id of the newly added atom or -1 if it couldn't be added
     *
     * Add an Atom to the simulation box. The type of the Atoms needs to exist
     * before the Atom is added. The velocity of the newly added Atom is
     * undefined and needs to be set either directly or via
     * InitializeAtoms::initTemp().
     *
     * \todo It would be nice to be able to set the velocity here.
     */
    int addAtom(std::string handle, Vector pos, double q);

    //! Directly add an Atom to the simulation
    /*!
     * \param a Atom to be added
     *
     * Directly add an Atom to the simulation. If the Atom mass is 0 or -1, the
     * mass will be determined from the Atom type. For 2d-simulations the
     * z-coordinate of the Atom will be set to zero.
     */
    bool addAtomDirect(Atom a);

    //! Remove an Atom from the simulation
    /*!
     * \param a Pointer to the Atom to be removed
     */
    bool deleteAtom(Atom *a);
    bool deleteMolecule(Molecule &);

    void createMolecule(std::vector<int> &ids);
    boost::python::object createMoleculePy(boost::python::list ids);
    void unwrapMolecules();

    boost::python::object duplicateMolecule(Molecule &, int n);
    Atom &duplicateAtom(Atom);
    void refreshIdToIdx();
    
    int nThreadPerAtom; //!< number of threads per atom for pair computations and nlist building
    int nThreadPerBlock; //!< number of threads per block for pair computations and nlist building
    int tuneEvery;
    
    bool verbose; //!< Verbose output
    int shoutEvery; //!< Report state of simulation every this many timesteps
    AtomParams atomParams; //!< Generic properties of the Atoms, e.g. masses,
                           //!< types, handles
    void findRigidBodies(); //!< Gets all rigid bodies associated with this simulation state
    bool rigidBodies; //!< Denotes whether rigid bodies are present in the simulation; 
    std::vector<int> rigidAtoms; //!< Boolean array that informs barostats whether the barostat performs the position rescaling (evaluates to true), else false (constraint algorithm handles the NPT position scaling and translation).

    GPUArrayGlobal<int> rigidBodiesMask;
    //!< Boolean mask for NPT simulations (otherwise unused) with rigid bodies; GPU side of rigidAtoms array.  False for rigid bodies (Barostat does /not/ handle the position rescaling; rather, the constraint algorithm does), true otherwise.
    // for now, let's keep this sorted by id?



    //! Return a copy of each Atom in the simulation
    /*!
     * \return Vector containing a copy of all Atoms in the simulation
     *
     * Returns a list of Atoms. These Atoms are copies of the Atoms in the
     * simulation and can be modified without changing the simulation state. To
     * modify the simulation, the list of Atoms can be modified and then passed
     * back to the simulation using State::setAtoms().
     */
    std::vector<Atom> copyAtoms();

    //! Set the list of Atoms
    /*!
     * \param fromSave List of Atoms to replace current Atoms
     *
     * Replace the current Atoms with a given list of Atoms. This could, for
     * example be all Atoms from a previous state saved via copyAtoms().
     */

    //! Delete all Atoms
    void deleteAtoms();

    //! Check whether a given Atom is in a given group
    /*!
     * \param a Reference of the given Atom
     * \param handle String specifying the group
     * \return True if Atom is in the group, else return False
     */
    bool atomInGroup(Atom &a, std::string handle);

    //! Perform asynchronous Host operations
    /*!
     * \param cb Function pointer for asynchronous calculation
     * \return Undefined
     *
     * This function copies all data to its respective buffer and opens a new
     * thread for an asynchronous operation. Typically this function is used
     * to write data to file and process Python operations.
     *
     * \todo Function does return neither True nor False
     */
    bool runtimeHostOperation(std::function<void (int64_t )> cb, bool async);

    boost::shared_ptr<std::thread> asyncData; //!< Shared pointer to a thread
    boost::shared_ptr<ReadConfig> readConfig; //!< Shared pointer to configuration reader

    //! Default constructor
    State();

    //! Set periodicity of the simulation box
    /*!
     * \param idx Index specifying the dimension (x: 0, y: 1, z: 2)
     * \param val True for periodic boundaries, False for fixed boundaries
     *
     * Set the periodicity of the simulation box for a given dimension.
     *
     * \todo Please check if the assert is correct here.
     */
    void setPeriodic(int idx, bool val) {
        assert(idx > 0 and idx < 3);
        periodic[idx] = val;
    }

    //! Return the periodicity of the simulation box
    /*!
     * \param idx Index specifying the dimension (x: 0, y: 1, z: 2)
     * \return True if boundary is periodic, False if it is fixed
     *
     * \todo Please check if the assert is correct here.
     */
    bool getPeriodic(int idx) {
        assert(idx > 0 and idx < 3);
        return periodic[idx];
    }

    //! Test whether Atom pointer is in atoms vector
    /*!
     * \param a Atom pointer to test
     * \return True if pointer is in the vector, else return False
     */
    bool validAtom(Atom *a);

    //! Refresh Fixes, Bonds, and Grid in case something's changed
    /*!
     * \return Always True
     *
     * In case Atoms or Groups have changed (indicated by the changedAtoms and
     * changedGroups member variables) or the neighborlist needs to be updated
     * (indicated by redoNeighbors), this function calls the necessary update
     * functions. The values of changedAtoms, changedGroups and redoNeighbors
     * are set back to False.
     *
     * \todo This function contains two commented out function calls:
     *       refreshBonds() and grid->periodicBoundaryConditions(). Either
     *       uncomment them or remove them if they are no longer necessary.
     */

    int maxExclusions; //!< Unused. \todo Remove? Grid uses maxExclusionsPerAtom

    //!
    void partitionAtoms();

    //! Copies atom data to gpu
    /*!
     * \return True always
     *
     * This function copies atom data to the gpu
     *
     */
    bool prepareForRun();
    void copyAtomDataToGPU(std::vector<int> &idToIdx);
    //! Prepares GridGPU member of state.  called after fix prepare run, because 
    /*!
     * \return True always
     *
     * This function copies atom data to the gpu
     *
     */


    //! Copy atom data from the GPU data back to the atom vectors
    /*!
     * \return True always
     *
     * Copy data from the GPU Data class back to the atoms vectors.
     */
    bool downloadFromRun();
//!resets various flags for fixes
    void finish(); 

    //! Set all Atom velocities to zero
    void zeroVelocities();

    //! Delete all Atoms and set pointers to NULL
    /*!
     * \todo This function feels out of place. First, it is confusing that State
     *       does not have an Destructor, but a destroy() function. Second,
     *       there are //UNCOMMENT comments for setting bounds pointers to NULL.
     *       Third, even when uncommented this is just one of many pointers that
     *       is set to NULL. I do not see the reason for this function.
     */
    void destroy();

    // these two are for managing atom ids such that they are densely packed
    // and it's quick at add atoms in large systems
    std::vector<int> idToIdx; //!< Cache for easier Atom index lookup.
    int idToIdxPy(int id); 

    Atom &idToAtom(int id);
    //! Maximum Atom Id for all existing Atoms
    /*!
     * The maximum Atom Id is not identical to the number of Atoms as Atoms can
     * be added and deleted.
     */
    int maxIdExisting;
    //! set gridGPU member.  used when preparing for run
    void initializeGrid();
    std::vector<int> idBuffer; //!< Buffer of unused Atom Ids

    //! Return reference to the Random Number Generator
    /*!
     * \return Random Number Generator
     *
     * Return a reference to the random number generator (RNG). If no seed has
     * been specified for the RNG, it is initialized with a random seed.
     *
     * \todo Maybe better pass a pointer to avoid accidental copying. For
     *       example std::mt19937 generator = state->getRNG() stores a copy,
     *       not a reference.
     *
     *       Returning by reference is bad, very bad. Is there a problem with
     *       coppying the RNG? You can return it as a reference_wrapper<T>
     *       maybe? It will force you to store it as a reference, then.
     */
    std::mt19937 &getRNG();

    //! Set the seed of the Random Number Generator
    /*!
     * \param seed Seed for the random number generator, 0 is random seed
     *
     * This function sets the seed for the random number generator (RNG). If the
     * seed is 0, the RNG is initialized with a random seed.
     */
    void seedRNG(unsigned int seed = 0);
    void handleChargeOffloading();

    Units units;

    bool preparePIMD(double temp);
    //! Extends current simulation state to a path-integral representation
    /*!
     * \return True always
     *
     * This function makes nPerRingPoly copies of the exiting atoms in the simulation
     * and their interactions for use in path-integral molecular dynamics simultions
     * currently, the same number of replicas are applied to all particles in the system.
     */

private:
    std::mt19937 randomNumberGenerator; //!< Random number generator
    bool rng_is_seeded; //!< True if seedRNG has been called
};

#endif

