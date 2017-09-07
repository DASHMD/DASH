#pragma once
#ifndef INTEGRATOR_UTIL_H
#define INTEGRATOR_UTIL_H
//so this class exists because integrators are not members of the class, but sometimes the state needs to internally call some things have to do with integration, like calculating energies.  
//The state has one of these classes.  Its methods are agnostic to integrator
class State;

class IntegratorUtil {
public:
    IntegratorUtil(State *);
    IntegratorUtil(){};
    State *state;
    //! Calculate force for all fixes
    /*!
     * \param computeVirials Compute virials for all forces if True
     *
     * This function iterates over all fixes and if the Fix should be applied
     * its force (and virials) is computed.
     *
     */
    void force(int virialMode);
    void postNVE_V();
    void postNVE_X();

    //! Collect data for all DataSets
    void doDataComputation();
    void doDataAppending();

    //! Calculate single point energy for all fixes
    /*!
     */
//    void singlePointEng(); 

    //! Calculate single point force
    /*!
     * \param computeVirials Virials are computed if this parameter is True
     *
     *
     */
    void forceSingle(int virialMode);
    void handleBoundsChange();

    void checkQuit();
};

#endif
