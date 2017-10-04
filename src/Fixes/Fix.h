#pragma once
#ifndef FIX_H
#define FIX_H

#include "Bond.h"
#include "globalDefs.h"
#include <pugixml.hpp>
#include <map>
#include "Interpolator.h"
#include "Tunable.h"

class Atom;
class State;
class EvaluatorWrapper;

//! Make class Fix available to Python interface
void export_Fix();

//! Base class for Fixes
/*!
 * Fixes modify the dynamics in the system. They are called by the Integrator
 * at regular intervals and they can modify the Atom forces, positions and
 * velocities. Note that as some Fixes depend on the current forces,
 * positions and velocities, the order in which the Fixes are defined and
 * called is important.
 *
 * For this reason, Fixes have an order preference. Fixes with a low preference
 * are computed first.
 *
 * \todo Compile list of preferences. I think each Fix should have an order
 *       preference.
 */
class Fix : public Tunable {

protected:
    //! Default constructor
    /*!
     * Delete default constructor.
     */
    Fix() = delete;

    //! Constructor
    /*!
     * \param state_ Pointer to simulation state
     * \param handle_ Name of the Fix
     * \param groupHandle_ String specifying on which group the Fix acts
     * \param type_ Type of the Fix (unused?)
     * \param applyEvery_ Apply Fix every this many timesteps
     *
     * \todo Make constructor protected since this is an abstract base class
     */
    Fix(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
        std::string type_, bool forceSingle_, bool requiresVirials_, bool requiresCharges_, int applyEvery_,
        int orderPreference_ = 0);
    boost::shared_ptr<EvaluatorWrapper> evalWrap;

public:
    //! Destructor
    virtual ~Fix() {};

    //! Test if another Fix is the same
    /*!
     * \param f Reference of Fix to test
     * \return True if they have the same handle
     *
     * Two Fixes are considered equal if they have the same "name" stored in
     * handle. Thus, a FixLJCut called "myFix" and a FixSpringStatic called
     * "myFix" are considered equal.
     *
     * \todo Take const reference, make function const
     * \todo Why not use operator==()?
     * \todo Isn't comparing the handle overly simplistic?
     */
    bool isEqual(Fix &f);

    //! Prepare Fix for run
    /*!
     * \return False if a problem occured, else True
     */
    virtual bool prepareForRun() {return true;};

    //! Perform calculations at the end of a simulation run
    /*!
     * \return False if an error occured, else return true
     *
     * Some Fixes set up internal variables in the Fix::prepareForRun()
     * function. This function then typically sets these values back to their
     * default.
     *
     * \todo Make this function purely virtual
     */
    virtual bool postRun() { prepared=false; return true; }

    //! Perform operations at the start of a simulation step
    /*!
     * \return False if a problem occured, else return true
     *
     * This function is called by the Integrator at the very start of each
     * simulation step.
     */
    virtual bool stepInit() { return true; }
    virtual bool postNVE_V() {return true; }
    virtual bool postNVE_X() {return true; } //postNVE_V and X are just called in first half step

    //! Prepares a fix for run if it must be prepared after all other fixes have been instantiated
    /*!
     * \return False if a problem occurred, else True
     *
     * This function is primarily useful for DataComputers and fixes that need to know the current 
     * state of the simulation in their absence (e.g., barostats, thermostats)
     */
    virtual bool prepareFinal() {return true; }

    //! Perform operations at the end of a simulation step
    /*!
     * \return False if a problem was encountered, else return true
     *
     * This function is called at the very end of a simulation step.
     */
    virtual bool stepFinal() { return true; }

    //! Counts the reduction in system DOF due to this fix
    /*!
     * \return 0 if no constraints, otherwise positive integer quantifying the reduction in DOF
     *
     * This function is called by DataComputerTemperature when it is preparing for a run.
     */
    virtual int removeNDF() {return 0;}
    //! Apply fix
    /*!
     * \param virialMode Compute virials for this Fix
     * \return False if a problem occured, else return true
     *
     * This function is called during the force calculation of the integration
     * step.
     */
    virtual void compute(int virialMode) {}

    //! Calculate single point energy of this Fix
    /*!
     * \param perParticleEng Pointer to where to store the per-particle energy
     *
     * The pointer passed needs to be a pointer to a memory location on the
     * GPU.
     *
     * \todo Use cuPointerGetAttribute() to check that the pointer passed is a
     *       pointer to GPU memory.
     *
     * \todo Make this function purely virtual.
     */
    virtual void singlePointEng(float *perParticleEng) {}
    virtual void singlePointEngGroupGroup(float *perParticleEng, uint32_t groupTagA, uint32_t groupTagB) {}

    //! Accomodate for new type of Atoms added to the system
    /*!
     * \param handle String specifying the new type of Atoms
     *
     * \todo Make purely virtual.
     */
    virtual void addSpecies(std::string handle) {}

    //! Recalculate group bitmask from a (possibly changed) handle
    void updateGroupTag();
    bool willFire(int64_t);//<!True if a fix will operate for the turn in the argument.

    //! Restart Fix
    /*!
     * \param restData XML node containing the restart data for the Fix
     *
     * \return False if restart data could not be loaded, else return True
     *
     * Setup Fix from restart data.
     */
    virtual bool readFromRestart(){return true;};//pugi::xml_node restData){return true;};
    pugi::xml_node getRestartNode();
   //! Makes copies of appropriate data to handle duplicating molecules
    /*!
     * \param map of ids - original to copied
     *
     * \return void
     *
     */
    virtual void duplicateMolecule(std::vector<int> &oldIds, std::vector<std::vector<int> > &newIds) {};

    virtual void deleteAtom(Atom *a) {};

    virtual void handleBoundsChange() {};
    //! Adjust any parameters that might need to be changed before compute
    /*!
     *
     *This would be used for interdependent fixes, like pair and charge.  Alpha parameter changes when bounds change, so evaluator that the pair fix has needs to be reset
     *
     */
    //! Write restart data
    /*!
     * \param format Format for restart data
     *
     * \return Restart string
     *
     * Write out information of this Fix to be reloaded via
     * Fix::readFromRestart().
     */
    virtual std::string restartChunk(std::string format){return "";};

    //! Return list of Bonds
    /*!
     * \return Pointer to list of Bonds or nullptr if Fix does not handle Bonds
     *
     * \todo Think about treatment of different kinds of bonds in fixes right
     *       now for ease, each vector of bonds in any given fix that stores
     *       bonds has to store them in a vector<BondVariant> variable you can
     *       push_back, insert, whatever, other kinds of bonds into this vector
     *       you have to get them out using a getBond method, or using the
     *       boost::get<BondType>(vec) syntax. It's not perfect, but it lets us
     *       generically collect vectors without doing any copying.
     */
    virtual std::vector<BondVariant> *getBonds() {
        return nullptr;
    }

    //! Return list of cutoff values.
    /*!
     * \return vector storing interaction cutoff values or empty list if no
     *         cutoffs are used.
     */
    virtual std::vector<float> getRCuts() {
        return std::vector<float>();
    }


    //! Returns the atom ids of atoms belonging to rigid bodies as denoted by this Fix.
    /*!
     * \return vector containing atom ids of atoms belonging to rigid bodies, or empty list if not 
     *         applicable to this fix (non-constraint fix).
     */

    virtual std::vector<int> getRigidAtoms() {
        return std::vector<int>();
    }

    virtual void scaleRigidBodies(float3 scaleBy, uint32_t groupTag) {};

    //! Check that all given Atoms are valid
    /*!
     * \param atoms List of Atom pointers
     *
     * This function verifies that all Atoms to be tested are valid using the
     * State::validAtom() method. The code crashes if an invalid Atom is
     * encountered.
     *
     * \todo A crash is not a very graceful method of saying that an Atom was
     *       invalid.
     * \todo Pass const reference. Make this function const.
     */
    void validAtoms(std::vector<Atom *> &atoms);
    //okay, so there's a bit of a structure going on with these evaluators.  
    //So each fix that can evaluate pair potential has an evaluator wrapper.
    //For the sake of efficiency, certain fixes (charges, at current) can offload their evaluators to other pair fixes.  
    //This can be seen in handleChargeOffloading in State.cpp.  After the charge fix is offloaded, the evaluator is set by the fix in prepareForRun.
    //There's a hitch though.  When I want to calculate per-fix energies, the charge fix needs to take back its evaluator so that short-range pair energies belong to that fix. 
    //The un-adulterated evaluator is origEvalWrapper, and that is used to calculate per-particle energies
    virtual void acceptChargePairCalc(Fix *){};

    virtual void setEvalWrapper(){};


    std::string evalWrapperMode;
    //self or offload
    void setEvalWrapperMode(std::string mode);

    State *state; //!< Pointer to the simulation state
    std::string handle; //!< "Name" of the Fix
    std::string groupHandle; //!< Group to which the Fix applies
    const std::string type; //!< String naming the Fix type
    int applyEvery; //!< Applyt this fix every this many timesteps
    unsigned int groupTag; //!< Bitmask for the group handle
    const bool forceSingle; //!< True if Fix contributes to single point energy.
    bool requiresVirials; //!< True if Fix needs virials.  Fixes will compute virials if any fix has this as true
    bool requiresPerAtomVirials; //!< True if Fix needs perAtom virials.  Fixes will compute per atom virials if any fix has this as true
    bool requiresCharges; //!< True if Fix needs charges.  Fixes will be stored if any fix has this as true
    //these are 
    bool isThermostat; //!< True if is a thermostat. Used for barostats.
    bool requiresForces; //!< True if the fix requires forces on instantiation; defaults to false.
    bool requiresPostNVE_V;

    bool prepared; //!< True if the fix has been prepared; false otherwise.
    bool canOffloadChargePairCalc;
    bool canAcceptChargePairCalc;
    
    bool hasOffloadedChargePairCalc;
    bool hasAcceptedChargePairCalc;
    void resetChargePairFlags();

    int orderPreference; //!< Fixes with a high order preference are calculated
                         //!< later.

    const std::string restartHandle; //!< Handle for restart string

    void setVirialTurnPrepare();
    void setVirialTurn();

    virtual Interpolator *getInterpolator(std::string);

    virtual void updateForPIMD(int nPerRingPoly) {};

    virtual void takeStateNThreadPerBlock(int);
    virtual void takeStateNThreadPerAtom(int);

};

#endif
