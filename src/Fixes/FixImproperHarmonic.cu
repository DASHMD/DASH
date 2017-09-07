#include "helpers.h"
#include "FixImproperHarmonic.h"
#include "FixHelpers.h"
#include "cutils_func.h"
#define SMALL 0.001f
#include "ImproperEvaluate.h"
namespace py = boost::python;
using namespace std;

const std::string improperHarmonicType = "ImproperHarmonic";


FixImproperHarmonic::FixImproperHarmonic(SHARED(State) state_, string handle)
    : FixPotentialMultiAtom (state_, handle, improperHarmonicType, true) {
        readFromRestart();

}


void FixImproperHarmonic::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    //printf("HELLO\n");
    if (forcersGPU.size()) {
        if (virialMode) {
            compute_force_improper<ImproperHarmonicType, ImproperEvaluatorHarmonic, true> <<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);

        } else {
            compute_force_improper<ImproperHarmonicType, ImproperEvaluatorHarmonic, false> <<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        }
    }
}
void FixImproperHarmonic::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    if (forcersGPU.size()) {
        compute_energy_improper<<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), usingSharedMemForParams, evaluator);
    }

}

void FixImproperHarmonic::createImproper(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq, int type) {
    vector<Atom *> atoms = {a, b, c, d};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and thetaEq!=COEF_DEFAULT);
    }
    forcers.push_back(ImproperHarmonic(a, b, c, d, k, thetaEq, type));
    pyListInterface.updateAppendedMember();
}
void FixImproperHarmonic::setImproperTypeCoefs(int type, double k, double thetaEq) {
    assert(thetaEq>=0);
    ImproperHarmonic dummy(k, thetaEq, type);
    setForcerType(type, dummy);
}


bool FixImproperHarmonic::readFromRestart() {
    auto restData = getRestartNode();
    if (restData) {
        auto curr_node = restData.first_child();
        while (curr_node) {
            std::string tag = curr_node.name();
            if (tag == "types") {
                for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
                    int type;
                    double k;
                    double thetaEq;
                    std::string type_ = type_node.attribute("id").value();
                    type = atoi(type_.c_str());
                    std::string k_ = type_node.attribute("k").value();
                    std::string thetaEq_ = type_node.attribute("thetaEq").value();
                    k = atof(k_.c_str());
                    thetaEq = atof(thetaEq_.c_str());

                    setImproperTypeCoefs(type, k, thetaEq);
                }
            } else if (tag == "members") {
                for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
                    int type;
                    double k;
                    double thetaEq;
                    int ids[4];
                    std::string type_ = member_node.attribute("type").value();
                    std::string atom_a = member_node.attribute("atomID_a").value();
                    std::string atom_b = member_node.attribute("atomID_b").value();
                    std::string atom_c = member_node.attribute("atomID_c").value();
                    std::string atom_d = member_node.attribute("atomID_d").value();
                    std::string k_ = member_node.attribute("k").value();
                    std::string thetaEq_ = member_node.attribute("thetaEq").value();
                    type = atoi(type_.c_str());
                    ids[0] = atoi(atom_a.c_str());
                    ids[1] = atoi(atom_b.c_str());
                    ids[2] = atoi(atom_c.c_str());
                    ids[3] = atoi(atom_d.c_str());
                    Atom * a = &state->idToAtom(ids[0]);
                    Atom * b = &state->idToAtom(ids[1]);
                    Atom * c = &state->idToAtom(ids[2]);
                    Atom * d = &state->idToAtom(ids[3]);
                    k = atof(k_.c_str());
                    thetaEq = atof(thetaEq_.c_str());

                    createImproper(a, b, c, d, k, thetaEq, type);
                }
            }
            curr_node = curr_node.next_sibling();
        }
    }
    return true;
}


void export_FixImproperHarmonic() {

    boost::python::class_<FixImproperHarmonic,
                          SHARED(FixImproperHarmonic),
                          boost::python::bases<Fix, TypedItemHolder> > (
        "FixImproperHarmonic",
        boost::python::init<SHARED(State), string> (
                boost::python::args("state", "handle"))
    )
    .def("createImproper", &FixImproperHarmonic::createImproper,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setImproperTypeCoefs", &FixImproperHarmonic::setImproperTypeCoefs,
            (boost::python::arg("type")=COEF_DEFAULT,
             boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("thetaEq")=COEF_DEFAULT
             )
        )

    .def_readonly("impropers", &FixImproperHarmonic::pyForcers)    
    ;


}

