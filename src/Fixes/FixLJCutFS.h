#pragma once
#ifndef FIXLJCUTFS_H
#define FIXLJCUTFS_H

#include "FixPair.h"
#include "PairEvaluatorLJFS.h"
#include "xml_func.h"

void export_FixLJCutFS();

//! Fix for truncated Lennard-Jones interactions
/*!
 * Fix to calculate Force shifted Lennard-Jones interactions of particles. 
 * Original LJ V(r)=4*eps*((sig/r)^12-(sig/r)^6)
 * Original LJ F(r)=24*eps*(2*(sig/r)^12-(sig/r)^6)*1/r
 * FS LJ F_fs(r)=F(r)-F(r_cut)
 */

class FixLJCutFS : public FixPair {
    public:
        //! Constructor
        FixLJCutFS(SHARED(State), std::string handle, std::string mixingRules_="geometric");

        //! Compute forces
        void compute(int);

        //! Compute single point energy
        void singlePointEng(float *);
        void singlePointEngGroupGroup(float *, uint32_t, uint32_t);

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
        const std::string epsHandle; //!< Handle for parameter epsilon
        const std::string sigHandle; //!< Handle for parameter sigma
        const std::string rCutHandle; //!< Handle for parameter rCut
        std::vector<float> epsilons; //!< vector storing epsilon values
        std::vector<float> sigmas; //!< vector storing sigma values
        std::vector<float> rCuts; //!< vector storing cutoff distance values
        std::vector<float> FCuts; //!< vector storing force at cutoff distance

        void setEvalWrapper();
};

#endif
