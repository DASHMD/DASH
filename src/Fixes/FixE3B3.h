#pragma once
#ifndef FIXE3B3_H
#define FIXE3B3_H

#include "globalDefs.h"
#include "FixPair.h"
#include "Fix.h"
#include "GPUArrayGlobal.h"

/* TODO: move pair interactions in to EvaluatorE3B3. */
//#include "PairEvaluatorE3B3.h"
#include "EvaluatorE3B3.h"
#include "GridGPU.h"
#include "Molecule.h"
#include "GPUData.h"

//! Make FixE3B3 available to the python interface
void export_FixE3B3();

//! Explicit 3-Body Potential, v3 (E3B3) for Water
/*
 * This fix implements the E3B3 water for water as 
 * described by Tainter, Shi, & Skinner in 
 * J. Chem. Theory Comput. 2015, 11, 2268-2277
 *
 * Note that this fix should only be used in conjunction 
 * with water modeled as TIP4P/2005
 */

class FixE3B3: public Fix {
    
    private:
   



    public:

        // delete the default constructor
        FixE3B3() = delete;

        /* FixE3B3 constructor
         * -- pointer to state
         * -- handle for the fix
         * -- group handle
         *
         *  In the constructor, we set the cutoffs required by this potential.
         */
        FixE3B3(boost::shared_ptr<State> state,
                  std::string handle,
                  std::string groupHandle);
        
        // far cutoff, rf = 5.2 Angstroms
        double rf;

        // short cutoff, rs = 5.0 Angstroms
        double rs;

        // cutoff of the neighborlist, rf + 2 Angstroms
        double rc;

        // implicitly defined by rc - rf = padding = 2.0 Angstroms
        double padding;

        //! Prepare Fix
        /*!
         * \returns Always returns True
         *
         * This function needs to be called before simulation run.
         */
        bool prepareForRun();

        //! Run after simulation
        /*!
         * This function needs to be called after simulation run.
         */

        bool stepInit();

        //void singlePointEng(float *);

        void compute(int);
        
        //! Reset parameters to before processing
        /*!
        * \param handle String specifying the parameter
        */
        //void handleBoundsChange();
        
        // we actually don't need the M-site for this..
        // but require it anyways, because this shouldonly be used with TIP4P/2005
        // -- takes atom IDs as O, H, H, M (see FixRigid.h, FixRigid.cu)
        void addMolecule(int, int, int, int);

        //void setEvalWrapper();
        //void setEvalWrapperOrig();
 
        //!< List of all water molecules in simulation
        std::vector<Molecule> waterMolecules;
       
        int nMolecules; // waterMolecules.size();
        //!< List of int4 atom ids for the list of molecules;
        //   The order of this list does /not/ change throughout the simulation
        GPUArrayDeviceGlobal<int4> waterIdsGPU;
        std::vector<int4> waterIds;
 
        // the local gridGPU for E3B3, where we make our molecule by molecule neighborlist
        GridGPU gridGPULocal;

        // corresponding local GPU data; note that we only really need xs - no need for fs, vs, etc..
        GPUData gpdLocal;

        // the evaluator for E3B3
        EvaluatorE3B3 evaluator;

};



#endif /* FIXE3B3_H */
