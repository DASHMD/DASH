#pragma once
#ifndef FIXTIP4PFLEXIBLE_H
#define FIXTIP4PFLEXIBLE_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include "Python.h"
#include "Fix.h"
#include "FixBond.h"
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"


void export_FixTIP4PFlexible();



/* FixTIP4PFlexible
 *
 * This fix implements flexible TIP4P models;
 * Essentially, it functions to partition the forces acting on the massless M-site
 * in a consistent manner throughout the simulation.
 *
 * The fix assumes that the initial configuration is such that all water molecules
 * are at the equilibrium geometry.
 *
 *
 *
 *
 * Supported 'styles' (where different styles usually denote different force partition constants) 
 * include
 *  - q-TIP4P/F: the default model (see reference Habershon, Markland , and Manolopoulos, J Chem. Phys. 131, 024501 (2009))
 *  - standard TIP4P/2005 geometry
 */
class FixTIP4PFlexible: public Fix {

    private:
    
        // holds the atom IDs associated with a given water molecule (including the M-site)
        GPUArrayDeviceGlobal<int4> waterIdsGPU;

        void compute_gamma();

        float gamma;

        std::vector<int4> waterIds;

        int nMolecules;
  
        std::vector<BondVariant> bonds;

        bool firstPrepare;

        void setStyleBondLengths(); 

    public:
        
        // constructor
        FixTIP4PFlexible(SHARED(State), std::string handle_);

        // prepareForRun
        bool prepareForRun();

        // stepFinal
        bool stepFinal();

        // handleBoundsChange
        void handleBoundsChange();
    
        // add a molecule to the fix
        void addMolecule(int, int, int, int);

        int removeNDF();

        // list of the bonds in this fix
        std::vector<BondVariant> *getBonds() {
            return &bonds;
        }
        
        std::string style;

        bool readFromRestart();

        std::string restartChunk(std::string format);
        // allows specification of a specific TIP4P flexible model geometry
        void setStyle(std::string);

        // alternatively, these are available directly from the python interface.
        double rOH;
        double rHH;
        double rOM;
        double theta;

        // see FixBond.h for more details
        void updateForPIMD(int nPerRingPoly);
};

#endif /* FIXTIP4PFLEXIBLE */
