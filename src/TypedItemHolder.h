#pragma once
#ifndef TYPEDITEMHOLDER_H
#include <vector>
/*! \class TypedItemHolder
 * \brief interface for extracting information about types of forcers held by fixes
 *
 * For example, Bond Harmonic, Angle Harmonic, etc will inherit this class, and define the apropriate methods to get #types existing, list used in the python LAMMPS reader
 * 
 */
void export_TypedItemHolder();
class TypedItemHolder {
    public:
        virtual std::vector<int> getTypeIds() {return std::vector<int>();}; //should be abstract, but boost does not really like that

};

#endif
