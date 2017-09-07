#include "helpers.h"
#include "FixDihedralOPLS.h"
#include "FixHelpers.h"
#include "cutils_func.h"
#include "DihedralEvaluate.h"
namespace py = boost::python;
using namespace std;

const std::string dihedralOPLSType = "DihedralOPLS";


FixDihedralOPLS::FixDihedralOPLS(SHARED(State) state_, string handle) : FixPotentialMultiAtom (state_, handle, dihedralOPLSType, true){
    readFromRestart();
}


void FixDihedralOPLS::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();


    GPUData &gpd = state->gpd;
    if (forcersGPU.size()) {
        if (virialMode) {
            compute_force_dihedral<DihedralOPLSType, DihedralEvaluatorOPLS, true><<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams >>>(forcersGPU.size(), gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        } else {
            compute_force_dihedral<DihedralOPLSType, DihedralEvaluatorOPLS, false><<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams >>>(forcersGPU.size(), gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        }
    }

}

void FixDihedralOPLS::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();

    GPUData &gpd = state->gpd;
    if (forcersGPU.size()) {
        compute_energy_dihedral<<<NBLOCK(forcersGPU.size()), PERBLOCK, sharedMemSizeForParams>>>(forcersGPU.size(), gpd.xs(activeIdx), perParticleEng, gpd.idToIdxs.d_data.data(), forcersGPU.data(), state->boundsGPU, parameters.data(), parameters.size(), usingSharedMemForParams, evaluator);
    }

}



void FixDihedralOPLS::createDihedral(Atom *a, Atom *b, Atom *c, Atom *d, double v1, double v2, double v3, double v4, int type) {
    double vs[4] = {v1, v2, v3, v4};
    if (type==-1) {
        for (int i=0; i<4; i++) {
            assert(vs[i] != COEF_DEFAULT);
        }
    }
    forcers.push_back(DihedralOPLS(a, b, c, d, vs, type));
    pyListInterface.updateAppendedMember();
}


void FixDihedralOPLS::createDihedralPy(Atom *a, Atom *b, Atom *c, Atom *d, py::list coefs, int type) {
    double coefs_c[4];
    if (type!=-1) {
        createDihedral(a, b, c, d, COEF_DEFAULT, COEF_DEFAULT, COEF_DEFAULT, COEF_DEFAULT, type);
    } else {
        assert(len(coefs) == 4);
        for (int i=0; i<4; i++) {
            py::extract<double> coef(coefs[i]);
            assert(coef.check());
            coefs_c[i] = coef;
        }
        createDihedral(a, b, c, d, coefs_c[0], coefs_c[1], coefs_c[2], coefs_c[3], type);

    }
}

void FixDihedralOPLS::setDihedralTypeCoefs(int type, py::list coefs) {
    assert(len(coefs)==4);
    double coefs_c[4];
    for (int i=0; i<4; i++) {
        py::extract<double> coef(coefs[i]);
        assert(coef.check());
        coefs_c[i] = coef;
    }

    DihedralOPLS dummy(coefs_c, type);
    setForcerType(type, dummy);
}

bool FixDihedralOPLS::readFromRestart() {
    auto restData = getRestartNode();
    if (restData) {
        auto curr_node = restData.first_child();
        while (curr_node) {
            string tag = curr_node.name();
            if (tag == "types") {
                for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
                    int type;
                    double coefs[4];
                    std::string type_ = type_node.attribute("id").value();
                    type = atoi(type_.c_str());
                    std::string coef_a = type_node.attribute("coef_a").value();
                    std::string coef_b = type_node.attribute("coef_b").value();
                    std::string coef_c = type_node.attribute("coef_c").value();
                    std::string coef_d = type_node.attribute("coef_d").value();
                    coefs[0] = atof(coef_a.c_str());
                    coefs[1] = atof(coef_b.c_str());
                    coefs[2] = atof(coef_c.c_str());
                    coefs[3] = atof(coef_d.c_str());
                    DihedralOPLS dummy(coefs, type);
                    setForcerType(type, dummy);
                }
            } else if (tag == "members") {
                for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
                    int type;
                    double coefs[4];
                    int ids[4];
                    std::string type_ = member_node.attribute("type").value();
                    std::string atom_a = member_node.attribute("atomID_a").value();
                    std::string atom_b = member_node.attribute("atomID_b").value();
                    std::string atom_c = member_node.attribute("atomID_c").value();
                    std::string atom_d = member_node.attribute("atomID_d").value();
                    std::string coef_a = member_node.attribute("coef_a").value();
                    std::string coef_b = member_node.attribute("coef_b").value();
                    std::string coef_c = member_node.attribute("coef_c").value();
                    std::string coef_d = member_node.attribute("coef_d").value();
                    type = atoi(type_.c_str());
                    ids[0] = atoi(atom_a.c_str());
                    ids[1] = atoi(atom_b.c_str());
                    ids[2] = atoi(atom_c.c_str());
                    ids[3] = atoi(atom_d.c_str());
                    coefs[0] = atof(coef_a.c_str());
                    coefs[1] = atof(coef_b.c_str());
                    coefs[2] = atof(coef_c.c_str());
                    coefs[3] = atof(coef_d.c_str());
                    Atom * a = &state->idToAtom(ids[0]);
                    Atom * b = &state->idToAtom(ids[1]);
                    Atom * c = &state->idToAtom(ids[2]);
                    Atom * d = &state->idToAtom(ids[3]);
                    if (a == NULL) {cout << "The first atom does not exist" <<endl; return false;};
                    if (b == NULL) {cout << "The second atom does not exist" <<endl; return false;};
                    if (c == NULL) {cout << "The third atom does not exist" <<endl; return false;};
                    if (d == NULL) {cout << "The fourth atom does not exist" <<endl; return false;};
                    createDihedral(a, b, c, d, coefs[0], coefs[1], coefs[2], coefs[3], type);
                }
            }
            curr_node = curr_node.next_sibling();
        }
    }
    return true;
}


void export_FixDihedralOPLS() {
    py::class_<FixDihedralOPLS,
                          SHARED(FixDihedralOPLS),
                          py::bases<Fix, TypedItemHolder> > (
        "FixDihedralOPLS",
        py::init<SHARED(State), string> (
            py::args("state", "handle")
        )
    )
    .def("createDihedral", &FixDihedralOPLS::createDihedralPy,
            (py::arg("coefs")=py::list(),
             py::arg("type")=-1)
        )

    .def("setDihedralTypeCoefs", &FixDihedralOPLS::setDihedralTypeCoefs, 
            (py::arg("type"), 
             py::arg("coefs"))
            )
    .def_readonly("dihedrals", &FixDihedralOPLS::pyForcers)

    ;

}

