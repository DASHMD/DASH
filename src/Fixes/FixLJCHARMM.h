#pragma once

#include "FixPair.h"
#include "xml_func.h"
//! Make FixLJCut available to the pair base class in boost
class EvaluatorWrapper;
void export_FixLJCHARMM();

//! Fix for CHARMM Lennard-Jones interactions
/*!
 * Fix to calculate Lennard-Jones interactions of particles. The LJ potential
 * is defined as
 * \f[
 * V(r_{ij}) = 4 \varepsilon \left[ \left(\frac{\sigma}{r_{ij}}\right)^{12} -
 *                               \left(\frac{\sigma}{r_{ij}}\right)^{6}\right],
 * \f]
 * where \f$ r \f$ is the distance between two particles and \f$ \varepsilon \f$
 * and \f$ \sigma \f$ are the two relevant parameters. The LJ pair interaction
 * is only calculated for particles closer than \$r_{\text{cut}}\$.
 *
 * From the potential, the force can be derived as
 * \f[
 * F(r_{ij}) = 24 \varepsilon \frac{1}{r} \left[ 
 *                          2 \left(\frac{\sigma}{r_{ij}}\right)^{12} -
 *                            \left(\frac{\sigma}{r_{ij}}\right)^{6}
 *                          \right].
 * \f]
 * If \f$F(r_{ij}) < 0\f$, then the force is attractive. Otherwise, it is
 * repulsive.
 */

extern const std::string LJCHARMMType;
class FixLJCHARMM : public FixPair {
    public:
        //! Constructor
        FixLJCHARMM(SHARED(State), std::string handle, std::string mixingRules_="geometric");

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
        void setEvalWrapper();
        const std::string epsHandle; //!< Handle for parameter epsilon
        const std::string sigHandle; //!< Handle for parameter sigma
        const std::string eps14Handle; //!< Handle for parameter epsilon for 1-4 interactions
        const std::string sig14Handle; //!< Handle for parameter sigma for 1-4 interactions
        const std::string rCutHandle; //!< Handle for parameter rCut
        std::vector<float> epsilons; //!< vector storing epsilon values
        std::vector<float> sigmas; //!< vector storing sigma values
        std::vector<float> epsilons14; //!< vector storing epsilon 1-4 values
        std::vector<float> sigmas14; //!< vector storing sigma 1-4 values
        std::vector<float> rCuts; //!< vector storing cutoff distance values


};

