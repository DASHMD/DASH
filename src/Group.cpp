#include "Group.h"
#include "State.h"
#include "boost_for_export.h"
#include <vector>
#include <string.h>
#include <iostream>
#include <stdint.h>

namespace py = boost::python;

using std::cout;
using std::endl;

Group::Group(State *state_, std::string groupHandle_) {
    state = state_;
    groupHandle = groupHandle_;
    groupTag = state->groupTagFromHandle(groupHandle);
    ndf = 0;
}


int Group::getNDF() {
    return ndf;
}


void Group::computeNDF() {
    ndf = 0; // we are about to sum over all atoms; re-set to zero
    for (Atom &a: state->atoms) {
        if (a.groupTag & groupTag) {
            ndf += a.ndf;
        }
    }
//    cout << "Group " << groupHandle << " was found to have " << ndf << " degrees of freedom." << endl;
}


// note that this will only yield an accurate NDF on 
// the python side /after/ a run command is issued, and it will 
// be incorrect if atoms are added to a group between runs, and the ndf is queried prior to issuing a 
// new run command.



void export_Group() { 
    py::class_<Group>("Group", py::no_init)
        .def_readonly("ndf", &Group::ndf)
        .def_readonly("groupHandle", &Group::groupHandle)
        //.add_property("pos", &Atom::getPos, &Atom::setPos)
    ;
}
