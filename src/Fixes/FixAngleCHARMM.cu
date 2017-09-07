
#include "FixHelpers.h"
#include "helpers.h"
#include "FixAngleCHARMM.h"
#include "cutils_func.h"
#include "AngleEvaluate.h"
using namespace std;
const string angleCHARMMType = "AngleCHARMM";
FixAngleCHARMM::FixAngleCHARMM(boost::shared_ptr<State> state_, string handle)
  : FixPotentialMultiAtom(state_, handle, angleCHARMMType, true)
{
    readFromRestart(); 
}

namespace py = boost::python;

void FixAngleCHARMM::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    GPUData &gpd = state->gpd;
    if (forcersGPU.size()) {
        if (virialMode) {
            compute_force_angle<AngleCHARMMType, AngleEvaluatorCHARMM, true> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + sharedMemSizeForParams>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        } else {
            compute_force_angle<AngleCHARMMType, AngleEvaluatorCHARMM, false> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + sharedMemSizeForParams>>>(nAtoms, gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        }
    }

}

void FixAngleCHARMM::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    if (forcersGPU.size()) {
        compute_energy_angle<<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + sharedMemSizeForParams>>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.d_data.data(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), usingSharedMemForParams, evaluator);
    }
}

void FixAngleCHARMM::createAngle(Atom *a, Atom *b, Atom *c, double k, double theta0, double kub, double rub,int type) {
    vector<Atom *> atoms = {a, b, c};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and theta0!=COEF_DEFAULT);
    }
    forcers.push_back(AngleCHARMM(a, b, c, k, theta0,kub,rub, type));
    pyListInterface.updateAppendedMember();
}

void FixAngleCHARMM::setAngleTypeCoefs(double k, double theta0, double kub, double rub, int type) {
    //cout << type << " " << k << " " << theta0 << endl;
    mdAssert(theta0>=0 and theta0 <= M_PI, "Angle theta must be between zero and pi" );
    mdAssert(rub>=0 , "Urey-Bradley distance, rub, must be above zero"  );
    AngleCHARMM dummy(k, theta0, kub, rub);
    setForcerType(type, dummy);
}


bool FixAngleCHARMM::readFromRestart() {
    auto restData = getRestartNode();
    if (restData) {
        auto curr_node = restData.first_child();
        while (curr_node) {
            std::string tag = curr_node.name();
            if (tag == "types") {
                for (auto type_node = curr_node.first_child(); type_node; type_node = type_node.next_sibling()) {
                    int type;
                    double k;
                    double theta0;
                    double kub;
                    double rub;
                    std::string type_ = type_node.attribute("id").value();
                    type = atoi(type_.c_str());
                    std::string k_ = type_node.attribute("k").value();
                    std::string theta0_ = type_node.attribute("theta0").value();
                    std::string kub_ = type_node.attribute("kub").value();
                    std::string rub_ = type_node.attribute("rub").value();
                    k = atof(k_.c_str());
                    theta0 = atof(theta0_.c_str());
                    kub = atof(kub_.c_str());
                    rub = atof(rub_.c_str());

                    setAngleTypeCoefs(k, theta0,kub,rub, type);
                }
            } else if (tag == "members") {
                for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
                    int type;
                    double k;
                    double theta0;
                    double kub;
                    double rub;
                    int ids[3];
                    std::string type_ = member_node.attribute("type").value();
                    std::string atom_a = member_node.attribute("atomID_a").value();
                    std::string atom_b = member_node.attribute("atomID_b").value();
                    std::string atom_c = member_node.attribute("atomID_c").value();
                    std::string k_      = member_node.attribute("k").value();
                    std::string theta0_ = member_node.attribute("theta0").value();
                    std::string kub_    = member_node.attribute("kub").value();
                    std::string rub_    = member_node.attribute("rub").value();
                    type = atoi(type_.c_str());
                    ids[0] = atoi(atom_a.c_str());
                    ids[1] = atoi(atom_b.c_str());
                    ids[2] = atoi(atom_c.c_str());
                    Atom * a = &state->idToAtom(ids[0]);
                    Atom * b = &state->idToAtom(ids[1]);
                    Atom * c = &state->idToAtom(ids[2]);
                    k      = atof(k_.c_str());
                    theta0 = atof(theta0_.c_str());
                    kub    = atof(kub_.c_str());
                    rub    = atof(rub_.c_str());

                    createAngle(a, b, c, k, theta0, kub, rub, type);
                }
            }
            curr_node = curr_node.next_sibling();
        }
    }
    return true;
}

void export_FixAngleCHARMM() {
    boost::python::class_<FixAngleCHARMM,
                          boost::shared_ptr<FixAngleCHARMM>,
                          boost::python::bases<Fix, TypedItemHolder> >(
        "FixAngleCHARMM",
        boost::python::init<boost::shared_ptr<State>, string>(
                                boost::python::args("state", "handle"))
    )
    .def("createAngle", &FixAngleCHARMM::createAngle,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("theta0")=COEF_DEFAULT,
             boost::python::arg("kub")=COEF_DEFAULT,
             boost::python::arg("rub")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setAngleTypeCoefs", &FixAngleCHARMM::setAngleTypeCoefs,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("theta0")=COEF_DEFAULT,
             boost::python::arg("kub")=COEF_DEFAULT,
             boost::python::arg("rub")=COEF_DEFAULT,
             boost::python::arg("type")=-1
            )
        )
    .def_readonly("angles", &FixAngleCHARMM::pyForcers)
    ;
}

