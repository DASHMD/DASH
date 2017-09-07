#include "helpers.h"
#include "FixImproperCVFF.h"
#include "FixHelpers.h"
#include "cutils_func.h"
#define SMALL 0.001f
#include "ImproperEvaluate.h"
namespace py = boost::python;
using namespace std;

const std::string improperCVFFType = "ImproperCVFF";


FixImproperCVFF::FixImproperCVFF(SHARED(State) state_, string handle)
    : FixPotentialMultiAtom (state_, handle, improperCVFFType, true) {
        readFromRestart();

}


void FixImproperCVFF::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    if (forcersGPU.size()) {
        if (virialMode) {
            compute_force_improper<ImproperCVFFType, ImproperEvaluatorCVFF, true> <<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);

        } else {
            compute_force_improper<ImproperCVFFType, ImproperEvaluatorCVFF, false> <<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        }
    }
}
void FixImproperCVFF::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    if (forcersGPU.size()) {
        compute_energy_improper<<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), usingSharedMemForParams, evaluator);
    }

}

void FixImproperCVFF::createImproper(Atom *a, Atom *b, Atom *c, Atom *d, double k, int dParam, int n, int type) {
    vector<Atom *> atoms = {a, b, c, d};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and (dParam==1 or dParam==-1) and n!=COEF_DEFAULT);
    }
    forcers.push_back(ImproperCVFF(a, b, c, d, k, dParam, n, type));
    pyListInterface.updateAppendedMember();
}
void FixImproperCVFF::setImproperTypeCoefs(int type, double k, int d, int n) {
    ImproperCVFF dummy(k, d, n, type);
    setForcerType(type, dummy);
}


bool FixImproperCVFF::readFromRestart() {
    auto restData = getRestartNode();
    if (restData) {
        auto curr_node = restData.first_child();
        while (curr_node) {
            std::string tag = curr_node.name();
            if (tag == "types") {
                for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
                    int type;
                    double k;
                    int d;
                    int n;
                    std::string type_ = type_node.attribute("id").value();
                    type = atoi(type_.c_str());
                    std::string k_ = type_node.attribute("k").value();
                    std::string d_ = type_node.attribute("d").value();
                    std::string n_ = type_node.attribute("n").value();
                    k = atof(k_.c_str());
                    d = atof(d_.c_str());
                    n = atof(n_.c_str());

                    setImproperTypeCoefs(type, k, d, n);
                }
            } else if (tag == "members") {
                for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
                    int type;
                    double k;
                    int dParam;
                    int n;
                    int ids[4];
                    std::string type_ = member_node.attribute("type").value();
                    std::string atom_a = member_node.attribute("atomID_a").value();
                    std::string atom_b = member_node.attribute("atomID_b").value();
                    std::string atom_c = member_node.attribute("atomID_c").value();
                    std::string atom_d = member_node.attribute("atomID_d").value();

                    std::string k_ = member_node.attribute("k").value();
                    std::string d_ = member_node.attribute("d").value();
                    std::string n_ = member_node.attribute("n").value();
                    k = atof(k_.c_str());
                    dParam = atoi(d_.c_str());
                    n = atoi(n_.c_str());
                    type = atoi(type_.c_str());
                    ids[0] = atoi(atom_a.c_str());
                    ids[1] = atoi(atom_b.c_str());
                    ids[2] = atoi(atom_c.c_str());
                    ids[3] = atoi(atom_d.c_str());
                    Atom * a = &state->idToAtom(ids[0]);
                    Atom * b = &state->idToAtom(ids[1]);
                    Atom * c = &state->idToAtom(ids[2]);
                    Atom * d = &state->idToAtom(ids[3]);

                    createImproper(a, b, c, d, k, dParam, n, type);
                }
            }
            curr_node = curr_node.next_sibling();
        }
    }
    return true;
}


__host__ void export_FixImproperCVFF() {

    boost::python::class_<FixImproperCVFF,
                          SHARED(FixImproperCVFF),
                          boost::python::bases<Fix, TypedItemHolder> > (
        "FixImproperCVFF",
        boost::python::init<SHARED(State), string> (
                boost::python::args("state", "handle"))
    )
    .def("createImproper", &FixImproperCVFF::createImproper,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("d")=COEF_DEFAULT,
             boost::python::arg("n")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setImproperTypeCoefs", &FixImproperCVFF::setImproperTypeCoefs,
            (boost::python::arg("type")=COEF_DEFAULT,
             boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("d")=COEF_DEFAULT,
             boost::python::arg("n")=COEF_DEFAULT
             )
        )

    .def_readonly("impropers", &FixImproperCVFF::pyForcers)    
    ;


}

