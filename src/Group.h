#pragma once
#ifndef GROUP_H
#define GROUP_H

#include <vector>
#include <string.h>
#include "Vector.h"
#include "globalDefs.h"
#include <boost/python/list.hpp>


void export_Group();

class State;

class Group {
    
    public:

        // grouptag associated with this group
        uint32_t groupTag;

        // groupHandle associated with this group
        std::string groupHandle;

        // ndf associated with this group
        int ndf;

        int getNDF();

        void computeNDF(); 

        // pointer to simulation state
        State *state;

        // probably not worth it to store in memory all atoms associated with this group..
        // instead, we iterate over atoms to find the members of the group

        Group() {};

        // pointer to state, groupHandle; we assign the groupTag via the grouphandle in the 
        // actual constructor
        Group(State* state_, std::string groupHandle_);

        // at the time of instantiation, a group iterates over all atoms that it owns to compute its NDF

};


#endif /* GROUP_H */

