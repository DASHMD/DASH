#pragma once
#ifndef FIXWCA_H
#define FIXWCA_H

#include "FixPair.h"
#include "PairEvaluatorWCA.h"
#include "xml_func.h"

//! Make FixWCA available to the pair base class in boost
void export_FixWCA();

//! Weeks Chandler Andersen potential (WCA)
//! Fix for purely repulsive Lennard-Jones interactions

/*!
 * Fix to calculate Lennard-Jones cutted at minimum (2^(1/6)sigma) at shifted up by eps. 
 * Original LJ V(r)=4*eps*((sig/r)^12-(sig/r)^6)
 * Original LJ F(r)=24*eps*(2*(sig/r)^12-(sig/r)^6)*1/r
 * V_WCA(r)=V(r)+eps
 */

class FixWCA : public FixPair {
    public:
        //! Constructor
        FixWCA(SHARED(State), std::string handle, std::string mixingRules_="geometric");

        //! Compute forces
        void compute(int);

        //! Compute single point energy
        void singlePointEng(float *);

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
        bool postRun();

        //! Create restart string
        /*!
         * \param format Format of the pair parameters.
         *
         * \returns restart chunk string.
         */
        std::string restartChunk(std::string format);


        //! Add new type of atoms
        /*!
         * \param handle Not used
         *
         * This function adds a new particle type to the fix.
         */
        void addSpecies(std::string handle);

        //! Return list of cutoff values
        std::vector<float> getRCuts();

    public:
   
        bool setParameter(std::string param,
                          std::string handleA,
                          std::string handleB,
                          double val);

      
        const std::string epsHandle; //!< Handle for parameter epsilon
        const std::string sigHandle; //!< Handle for parameter sigma
        const std::string rCutHandle; //!< Handle for parameter rCut
        std::vector<float> epsilons; //!< vector storing epsilon values
        std::vector<float> sigmas; //!< vector storing sigma values
        std::vector<float> rCuts; //!< vector storing cutoff distance values

        void setEvalWrapper();
        void setEvalWrapperOrig();
};

#endif
