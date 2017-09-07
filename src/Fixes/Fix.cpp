#include "Fix.h"

#include <iostream>

#include "Atom.h"
#include "boost_for_export.h"
#include "list_macro.h"
#include "ReadConfig.h"
#include "State.h"

//requiresVirials is deprecated.  Should be removed.  Any virial stuff is handled via data computers.
Fix::Fix(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
         std::string type_, bool forceSingle_, bool requiresVirials_, bool requiresCharges_, int applyEvery_,
         int orderPreference_)
    : state(state_.get()), handle(handle_), groupHandle(groupHandle_),
      type(type_), forceSingle(forceSingle_), requiresVirials(requiresVirials_),
      requiresCharges(requiresCharges_), applyEvery(applyEvery_), isThermostat(false),
      orderPreference(orderPreference_), restartHandle(type + "_" + handle)
{
    updateGroupTag();
    requiresPostNVE_V = false;

    requiresPerAtomVirials = false;

    canOffloadChargePairCalc = false;
    canAcceptChargePairCalc = false;
    
    hasOffloadedChargePairCalc = false;
    hasAcceptedChargePairCalc = false;
    setEvalWrapperMode("offload"); //offload by default
    nThreadPerAtom(state->nThreadPerAtom);


    /*
     * implemented per-fix.  May need to initialize junk first
    if (state->readConfig->fileOpen) {
        auto restData = state->readConfig->readNode(restartHandle);
        if (restData) {
            std::cout << "Reading restart data for fix " << handle << std::endl;
            readFromRestart(restData);
        }
    }
    */
}

bool Fix::willFire(int64_t t) {
    return ! (t % applyEvery);
}

void Fix::setVirialTurnPrepare() {
    if (requiresVirials) {
        double multiple = ceil(state->turn / applyEvery);
        state->dataManager.addVirialTurn(multiple * applyEvery, requiresPerAtomVirials);
    }
}
void Fix::setVirialTurn() {
    if (requiresVirials) {
        state->dataManager.addVirialTurn(state->turn + applyEvery, requiresPerAtomVirials);
    }
}

void Fix::resetChargePairFlags() {

    hasOffloadedChargePairCalc = false;
    hasAcceptedChargePairCalc = false;
}
bool Fix::isEqual(Fix &f) {
    return f.handle == handle;
}

pugi::xml_node Fix::getRestartNode() {
    if (state->readConfig->fileOpen) {
        auto restData = state->readConfig->readFix(type, handle);
        return restData;
    }
    return pugi::xml_node();

}
void Fix::updateGroupTag() {
    std::map<std::string, unsigned int> &groupTags = state->groupTags;
    if (groupHandle == "None" or groupHandle == "none") {
        groupTag = 0;
    } else {
        assert(groupTags.find(groupHandle) != groupTags.end());
        groupTag = groupTags[groupHandle];
    }
}

void Fix::validAtoms(std::vector<Atom *> &atoms) {
    for (int i=0; i<atoms.size(); i++) {
        if (!state->validAtom(atoms[i])) {
            std::cout << "Tried to create for " << handle
                      << " but atom " << i << " was invalid" << std::endl;
            assert(false);
        }
    }
}

void Fix::setEvalWrapperMode(std::string mode) {
    if (mode == "offload") { 
        evalWrapperMode = mode;
    } else if (mode == "self") {
        evalWrapperMode = mode;
    } else {
        std::cout << "Invalid evaluator wrapper mode " << mode << ".  This is an internal error." << std::endl;
        assert(mode == "offload" or mode == "self");
    }
    
}


void Fix::takeStateNThreadPerBlock(int nThread) {
    nThreadPerBlock(nThread);

}
void Fix::takeStateNThreadPerAtom(int nThread) {
    nThreadPerAtom(nThread);
}
Interpolator *Fix::getInterpolator(std::string type) {
    return nullptr;
}

void export_Fix() {
    boost::python::class_<Fix, SHARED(Fix), boost::noncopyable> (
        "Fix",
        boost::python::no_init
    )
    .def("prepareForRun", &Fix::prepareForRun)
    .def_readonly("handle", &Fix::handle)
    .def_readonly("type", &Fix::type)
    .def_readwrite("applyEvery", &Fix::applyEvery)
    .def_readwrite("groupTag", &Fix::groupTag)
    ;

}

