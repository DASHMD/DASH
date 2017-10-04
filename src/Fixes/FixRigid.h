#pragma once
#ifndef FIXRIGID_H
#define FIXRIGID_H

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
#include "../Eigen/Dense"

void export_FixRigid();

// simple class holding static data associated with a given water model
// -- namely, masses & corresponding weights of hydrogens, oxygen;
//    and the geometry of the speciic model
class FixRigidData {
    public:
        // bond lengths - as OH1, OH2, HH, OM
        // SET IN : setStyleBondLengths(), called by prepareForRun()
        double4 sideLengths;


        // as 1.0 / sideLengths, element-wise
        // SET IN : setStyleBondLengths(), called by prepareForRun()
        double4 invSideLengths;

        // canonical lengths, with center of mass as origin
        // as (ra, rb, rc, inv2Rc)
        // SET IN:  set_fixed_sides(), in prepareForRun()
        double4 canonicalTriangle;

        // mass weights - make these double precision!
        // --- note - useful operator definitions were provided in cutils_func and cutils_math..
        //     --- should do this for double2, double3, double4 etc.
        // -- weights.x = massO / (massWater); 
        //    weights.y = massH / massWater;
        //    weights.z = massO;
        //    weights.w = massH;
        // SET IN : set_fixed_sides(), in prepareForRun();
        double4 weights;
        
        // 1.0 / weights, element-wise
        // SET IN : set_fixed_sides(), in prepareForRun();
        double4 invMasses;

        // three arrays for constraint coupling matrix, containing mass weights by which we 
        // multiply to solve the velocity constraints
        // and their inverses below
        // SET IN :  populateRigidData(), called by prepareForRun()
        double3 M1;
        double3 M2;
        double3 M3;

        // SET IN :  populateRigidData(), called by prepareForRun()
        double3 M1_inv;
        double3 M2_inv;
        double3 M3_inv;


        double inv2Rc;
        
        // for 4-site models, this defines the length along the bisector of the HOH angle that the M-site is positioned.
        // we use eq. 2 from Manolopoulos et. al. as the defining quantity
        // see J. Chem. Phys. 131, 024501 (2009) 
        double gamma;


        // and the constructor
        FixRigidData() {};
};


class FixRigid : public Fix {
    private:
        // array holding ids of a given water molecule
        GPUArrayDeviceGlobal<int4> waterIdsGPU;

        // array holding positions
        GPUArrayDeviceGlobal<float4> xs_0;

        GPUArrayDeviceGlobal<double3> velCorrectionStored;

        // array holding COM before constraints are applied..
        GPUArrayDeviceGlobal<float4> com;

        // vector of booleans that will alert us if a molecule has unsatisfied constraints at the end of the turn
        GPUArrayDeviceGlobal<bool> constraints;


        std::vector<int4> waterIds;


        std::vector<BondVariant> bonds;


        std::vector<float4> invMassSums;


        // boolean defaulting to false in the constructor, denoting whether this is TIP4P/2005
        bool TIP4P;

        // boolean defaulting to false in the constructor, denoting whether this is TIP3P
        bool TIP3P;


        int nMolecules;

        // local constants to be set for assorted supported water models
        // sigma_O is for scaling of bond lengths if LJ units are used
        double sigma_O;
        double r_OH;
        double r_HH;
        double r_OM;
        double gamma; 

        // a float 4 of the above measures r_OH, r_HH, r_OM;
        double4 fixedSides;

        // data about this water model
        FixRigidData fixRigidData;
        // 
        void set_fixed_sides();

        void setStyleBondLengths();
        // 
    public:


        //! Constructor
        /*!
        * \param state Pointer to the simulation state
        * \param handle "Name" of the Fix
        * \param groupHandle String specifying group of atoms this Fix acts on
        */
        FixRigid(SHARED(State), std::string handle_, std::string groupHandle_);


        //! First half step of the integration
        /*!
         * \return Result of the FixRigid::stepInit() call.
         */
        bool stepInit();

        //! Second half step of the integration
        /*!
         * \return Result of FixRigid::stepFinal() call.
         */
        bool stepFinal();

        bool postNVE_V();

        bool postNVE_X();
        // populate the FixRigidData instance
        void populateRigidData();

        //! Prepare FixRigid for simulation run
        bool prepareForRun();
      
        // reduces the NDF of atoms governed by this constraint
        void assignNDF();

        // the style of a given water model to be used.  Defaults to 'DEFAULT',  
        // either TIP3P or TIP4P/2005 depending on whether the model is 3-site or 4-site
        std::string style;

        // boolean defaulting to true in the constructor;
        // dictates whether or not we alter the initial configuration and call settlePositions & settleVelocities
        // --- in general, if the initial configuration is good, then calling settlePositions & settleVelocities 
        //     will have no effect
        bool solveInitialConstraints;

        // permits variants of a given style (e.g., TIP4P, TIP4P/LONG, TIP4P/2005)
        // default styles are TIP3P and TIP4P/2005
        void setStyle(std::string);

        std::string restartChunk(std::string format);

        bool readFromRestart();

        
        std::vector<int> getRigidAtoms();

        void scaleRigidBodies(float3 scaleBy, uint32_t groupTag);

        //! Halfstep solution to velocity constraints
        //bool postNVE_V();

        //! Reset the position of the M-site after integrating the position of the molecule.
        //  -- Note that we do /not/ solve the rigid body constraints at this point;
        //     However, the massless site does not evolve with the other atoms of the molecule (if TIP4P);
        //     So, manually re-set it here for accuracy.
        //     Also, check if this is rigorously correct or a valid approximation. \TODO/
        // void handleBoundsChange();

        //! Removes NDF corresponding to the number of water molecules in the fix
        int removeNDF();

        // Create a rigid constraint on a TIP3P water molecule
        /*!
         * \ param id_a The atom id in the simulation state of the Oxygen atom in TIP3P
         * \ param id_b The atom id in the simulation state of a Hydrogen atom in TIP3P
         * \ param id_c The atom id in the simulation state of a Hydrogen atom in TIP3P
         */
        void createRigid(int, int, int);


        bool postRun();
        //! Create a rigid constraint on a TIP4P/2005 water molecule
        /*!
         * \ param id_a The atom id in the simulation state of the Oxygen atom in TIP4P/2005
         * \ param id_b The atom id in the simulation state of a Hydrogen atom in TIP4P/2005
         * \ param id_c The atom id in the simulation state of a Hydrogen atom in TIP4P/2005
         * \ param id_d The atom id in the simulation state of the M site in TIP4P/2005
         */
        void createRigid(int, int, int, int);


        bool printing;
       

        bool firstPrepare;
        std::vector<BondVariant> *getBonds() {
            return &bonds;
  }
};


#endif /* FIXRIGID_H */
