#pragma once
#ifndef FIXTICG_H
#define FIXTICG_H

#include "FixPair.h"
#include "PairEvaluatorTICG.h"
#include "xml_func.h"

//! Make FixTICG available to the pair base class in boost
void export_FixTICG();

//! Fix for soft TICG interactions
/*!
 * Fix to calculate TICG interactions of particles. 
 * Energy is proportional to volume of overlapping spheres over the volume of the whole sphere 
 * rCut (or radius of interactions r_int) is 2xRadius of spheres
 * C is stregth of potential (i.e E=C*V_intersection/V_sphere)
 */
class FixTICG : public FixPair {
    public:
        //! Constructor
        FixTICG(SHARED(State), std::string handle, std::string mixingRules="geometric");

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
        const std::string CHandle; //!< Handle for parameter C -stregth of potential
        const std::string rCutHandle; //!< Handle for parameter rCut
        std::vector<float> Cs; //!< vector storing epsilon values
        std::vector<float> rCuts; //!< vector storing cutoff distance values

        void setEvalWrapper();
};

#endif
