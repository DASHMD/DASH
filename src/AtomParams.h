#pragma once
#ifndef ATOM_PARAMS_H
#define ATOM_PARAMS_H

#include <string>
#include <vector>

//herm, would like to allow for different force fields
//
//
//id maps to index of data in mass, sigmas, epsilons
//
//this class does not hold per-atom info
void export_AtomParams();

class State;

/*! \class AtomParams
 * \brief Class storing all available general info on atoms
 *
 * This class stores and manages all available, general info on atoms, such as
 * the number of atom types, their masses and Number in the periodic table.
 * This class does not store per-atom data.
 */
class AtomParams {
public:

    /*! \brief Default constuctor */
    AtomParams() : numTypes(0) {};

    /*! \brief constructor
     *
     * \param s Pointer to the corresponding state
     *
     * Constructor setting the pointer to the corresponding state.
     */
    AtomParams(State *s) : state(s), numTypes(0) {};

    /*! \brief Add a new type of atoms to the system
     *
     * \param handle Unique identifier for the atoms
     * \param mass Mass of the atoms
     * \param atomicNum Position in the periodic table
     * \returns -1 if handle already exists and updated number of atom
     *          types otherwise.
     *
     * Add a new type of atoms to the system.
     */
    int addSpecies(std::string handle, double mass, int atomicNum=6);

    /*! \brief Remove all atom type info
     *
     * Delete all info on atom types previously stored in this class.
     */
    void clear();

    /*! \brief Return atom type for a given handle
     *
     * \param handle Unique identifier for the atom type
     * \returns Integer specifying the atom type
     *
     * Get the atom type for a given handle.
     */
    int typeFromHandle(const std::string &handle) const;
    
    void setValues(std::string handle, double mass, double atomicNum);

public:
    State *state; //!< state class

    int numTypes; //!< Number of atom types
    std::vector<std::string> handles; //!< List of handles to specify atom types
    std::vector<double> masses; //!< List of masses, one for each atom type
    std::vector<int> atomicNums; //!< For each atom type, this vector stores
                                    //!< its number in the periodic table
    void guessAtomicNumbers();
};

#endif
