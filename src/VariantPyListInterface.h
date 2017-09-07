#pragma once
#ifndef VARIANTPYLISTINTERFACE_H
#define VARIENTPYLISTINTERFACE_H

#include <vector>

#include <boost/shared_ptr.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <boost/variant.hpp>

#include "Python.h"

/*! \brief Bond connecting atomsclass for exposing vectors of variants (bonds, angles, dihedrals, impropers) to the python api
 */

template <class CPUMember>
void deleter(CPUMember *ptr) { };

template <class CPUVariant, class CPUMember>
class VariantPyListInterface {

private:
    std::vector<CPUVariant> *CPUMembers;
    boost::python::list *pyList;
    CPUVariant *CPUData;
    void refreshPyList() {
        int ii = boost::python::len(*pyList);
        for (int i=0; i<ii; i++) {
            CPUMember *member = boost::get<CPUMember>(&(*CPUMembers)[i]);
            boost::shared_ptr<CPUMember> shrptr (member, deleter<CPUMember>);
            (*pyList)[i] = shrptr;
        }
    }
public:
   
    VariantPyListInterface(std::vector<CPUVariant> *CPUMembers_, boost::python::list *pyList_)
      : CPUMembers(CPUMembers_), pyList(pyList_), CPUData(CPUMembers->data())
    {   }

    void requestRefreshPyList(bool force=false) {
        if (CPUMembers->data() != CPUData || force) {
            refreshPyList();
            CPUData = CPUMembers->data();
        }
    }
 
    void updateAppendedMember(bool copy=true) {
        if (copy) {
            requestRefreshPyList();
        }
        CPUMember *member = boost::get<CPUMember>(&(CPUMembers->back()));
        boost::shared_ptr<CPUMember> shrptr(member, deleter<CPUMember>);
        pyList->append(shrptr);
    }
    void removeMember(int i) {
        pyList->pop(i);
    }

};

#endif
