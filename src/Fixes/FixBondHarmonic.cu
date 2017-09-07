#include "helpers.h"
#include "FixBondHarmonic.h"
#include "cutils_func.h"
#include "FixHelpers.h"
#include "BondEvaluate.h"
#include "ReadConfig.h"
namespace py = boost::python;
using namespace std;

const std::string bondHarmonicType = "BondHarmonic";

FixBondHarmonic::FixBondHarmonic(SHARED(State) state_, string handle)
    : FixBond(state_, handle, string("None"), bondHarmonicType, true, 1) {
        readFromRestart();
    }

//template <class BONDTYPE, class EVALUATOR, bool COMPUTEVIRIALS>
__global__ void dummy(){};//int nAtoms, float4 *xs, float4 *forces){}//, int *idToIdxs, BondGPU *bonds, int *startstops, BONDTYPE *parameters_arg, int nParameters, BoundsGPU bounds, Virial *__restrict__ virials, bool usingSharedMemForParams, EVALUATOR T) {}

void FixBondHarmonic::createBond(Atom *a, Atom *b, double k, double r0, int type) {
    vector<Atom *> atoms = {a, b};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=-1 and r0!=-1);
    }
    bonds.push_back(BondHarmonic(a, b, k, r0, type));
    pyListInterface.updateAppendedMember();
    
}

void FixBondHarmonic::setBondTypeCoefs(int type, double k, double r0) {
    assert(r0>=0);
    BondHarmonic dummy(k, r0, type);
    setBondType(type, dummy);
}

void FixBondHarmonic::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    GPUData &gpd = state->gpd;
    //cout << "Max bonds per block is " << maxBondsPerBlock << endl;
    if (bondsGPU.size()) {
        if (virialMode) {
            compute_force_bond<BondHarmonicType, BondEvaluatorHarmonic, true> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondGPU) * maxBondsPerBlock + sharedMemSizeForParams>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), bondsGPU.data(), bondIdxs.data(), parameters.data(), parameters.size(), state->boundsGPU, gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        } else {
            compute_force_bond<BondHarmonicType, BondEvaluatorHarmonic, false> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondGPU) * maxBondsPerBlock + sharedMemSizeForParams>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), bondsGPU.data(), bondIdxs.data(), parameters.data(), parameters.size(), state->boundsGPU, gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        }
    }
}

void FixBondHarmonic::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    //cout << "Max bonds per block is " << maxBondsPerBlock << endl;
    if (bondsGPU.size()) {
        compute_energy_bond<<<NBLOCK(nAtoms), PERBLOCK, sizeof(BondGPU) * maxBondsPerBlock + sharedMemSizeForParams>>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.d_data.data(), bondsGPU.data(), bondIdxs.data(), parameters.data(), parameters.size(), state->boundsGPU, usingSharedMemForParams, evaluator);
    }
}

string FixBondHarmonic::restartChunk(string format) {
    stringstream ss;
    ss << "<types>\n";
    for (auto it = bondTypes.begin(); it != bondTypes.end(); it++) {
        ss << "<" << "type id='" << it->first << "'";
        ss << bondTypes[it->first].getInfoString() << "'/>\n";
    }
    ss << "</types>\n";
    ss << "<members>\n";
    for (BondVariant &forcerVar : bonds) {
        BondHarmonic &forcer= boost::get<BondHarmonic>(forcerVar);
        ss << forcer.getInfoString();
    }
    ss << "</members>\n";
    return ss.str();
}

bool FixBondHarmonic::readFromRestart() {
    auto restData = getRestartNode();
    if (restData) {
        auto curr_node = restData.first_child();
        while (curr_node) {
            std::string tag = curr_node.name();
            if (tag == "types") {
                for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
                    int type;
                    double k;
                    double r0;
                    std::string type_ = type_node.attribute("id").value();
                    type = atoi(type_.c_str());
                    std::string k_ = type_node.attribute("k").value();
                    std::string r0_ = type_node.attribute("r0").value();
                    k = atof(k_.c_str());
                    r0 = atof(r0_.c_str());

                    setBondTypeCoefs(type, k, r0);
                }
            } else if (tag == "members") {
                for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
                    int type;
                    double k;
                    double r0;
                    int ids[2];
                    std::string type_ = member_node.attribute("type").value();
                    std::string atom_a = member_node.attribute("atomID_a").value();
                    std::string atom_b = member_node.attribute("atomID_b").value();
                    std::string k_ = member_node.attribute("k").value();
                    std::string r0_ = member_node.attribute("r0").value();
                    type = atoi(type_.c_str());
                    ids[0] = atoi(atom_a.c_str());
                    ids[1] = atoi(atom_b.c_str());
                    Atom * a = &state->idToAtom(ids[0]);
                    Atom * b = &state->idToAtom(ids[1]);
                    k = atof(k_.c_str());
                    r0 = atof(r0_.c_str());

                    createBond(a, b, k, r0, type);
                }
            }
            curr_node = curr_node.next_sibling();
        }
    }
    return true;
}

void export_FixBondHarmonic() {
  

  
    py::class_<FixBondHarmonic, SHARED(FixBondHarmonic), py::bases<Fix, TypedItemHolder> >
    (
        "FixBondHarmonic", py::init<SHARED(State), string> (py::args("state", "handle"))
    )
    .def("createBond", &FixBondHarmonic::createBond,
            (py::arg("k")=-1,
             py::arg("r0")=-1,
             py::arg("type")=-1)
        )
    .def("setBondTypeCoefs", &FixBondHarmonic::setBondTypeCoefs,
            (py::arg("type"),
             py::arg("k"),
             py::arg("r0"))
        )
    .def_readonly("bonds", &FixBondHarmonic::pyBonds)    
    ;

}
