
#include "FixHelpers.h"
#include "helpers.h"
#include "FixAngleCosineDelta.h"
#include "cutils_func.h"
#include "AngleEvaluate.h"
using namespace std;
const string angleCosineDeltaType = "AngleCosineDelta";
FixAngleCosineDelta::FixAngleCosineDelta(boost::shared_ptr<State> state_, string handle)
  : FixPotentialMultiAtom(state_, handle, angleCosineDeltaType, true)
{
    readFromRestart();

}

namespace py = boost::python;

void FixAngleCosineDelta::compute(int virialMode) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    GPUData &gpd = state->gpd;
    if (forcersGPU.size()) {
        if (virialMode) {
            compute_force_angle<AngleCosineDeltaType, AngleEvaluatorCosineDelta, true> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + sharedMemSizeForParams>>>(nAtoms, state->gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);
        } else {
            compute_force_angle<AngleCosineDeltaType, AngleEvaluatorCosineDelta, false> <<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + sharedMemSizeForParams>>>(nAtoms, state->gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), gpd.virials.d_data.data(), usingSharedMemForParams, evaluator);

        }
    }

}

void FixAngleCosineDelta::singlePointEng(float *perParticleEng) {
    int nAtoms = state->atoms.size();
    int activeIdx = state->gpd.activeIdx();
    if (forcersGPU.size()) {
        compute_energy_angle<<<NBLOCK(nAtoms), PERBLOCK, sizeof(AngleGPU) * maxForcersPerBlock + sharedMemSizeForParams >>>(nAtoms, state->gpd.xs(activeIdx), perParticleEng, state->gpd.idToIdxs.d_data.data(), forcersGPU.data(), forcerIdxs.data(), state->boundsGPU, parameters.data(), parameters.size(), usingSharedMemForParams, evaluator);
    }
}
//void cumulativeSum(int *data, int n);
// okay, so the net result of this function is that two arrays (items, idxs of
// items) are on the gpu and we know how many bonds are in bondiest block

void FixAngleCosineDelta::createAngle(Atom *a, Atom *b, Atom *c, double k, double theta0, int type) {
    vector<Atom *> atoms = {a, b, c};
    validAtoms(atoms);
    if (type == -1) {
        assert(k!=COEF_DEFAULT and theta0!=COEF_DEFAULT);
    }
    forcers.push_back(AngleCosineDelta(a, b, c, k, theta0, type));
    pyListInterface.updateAppendedMember();
}

void FixAngleCosineDelta::setAngleTypeCoefs(int type, double k, double theta0) {
    //cout << type << " " << k << " " << theta0 << endl;
    mdAssert(theta0>=0 and theta0 <= M_PI, "Angle theta must be between zero and pi");
    AngleCosineDelta dummy(k, theta0);
    setForcerType(type, dummy);
}


bool FixAngleCosineDelta::readFromRestart() {
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
                    std::string type_ = type_node.attribute("id").value();
                    type = atoi(type_.c_str());
                    std::string k_ = type_node.attribute("k").value();
                    std::string theta0_ = type_node.attribute("theta0").value();
                    k = atof(k_.c_str());
                    theta0 = atof(theta0_.c_str());

                    setAngleTypeCoefs(type, k, theta0);
                }
            } else if (tag == "members") {
                for (auto member_node = curr_node.first_child(); member_node; member_node = member_node.next_sibling()) {
                    int type;
                    double k;
                    double theta0;
                    int ids[3];
                    std::string type_ = member_node.attribute("type").value();
                    std::string atom_a = member_node.attribute("atomID_a").value();
                    std::string atom_b = member_node.attribute("atomID_b").value();
                    std::string atom_c = member_node.attribute("atomID_c").value();
                    std::string k_ = member_node.attribute("k").value();
                    std::string theta0_ = member_node.attribute("theta0").value();
                    type = atoi(type_.c_str());
                    ids[0] = atoi(atom_a.c_str());
                    ids[1] = atoi(atom_b.c_str());
                    ids[2] = atoi(atom_c.c_str());
                    Atom * a = &state->idToAtom(ids[0]);
                    Atom * b = &state->idToAtom(ids[1]);
                    Atom * c = &state->idToAtom(ids[2]);
                    k = atof(k_.c_str());
                    theta0 = atof(theta0_.c_str());

                    createAngle(a, b, c, k, theta0, type);
                }
            }
            curr_node = curr_node.next_sibling();
        }
    }
    return true;
}

void export_FixAngleCosineDelta() {
    boost::python::class_<FixAngleCosineDelta,
                          boost::shared_ptr<FixAngleCosineDelta>,
                          boost::python::bases<Fix, TypedItemHolder> >(
        "FixAngleCosineDelta",
        boost::python::init<boost::shared_ptr<State>, string>(
                                boost::python::args("state", "handle"))
    )
    .def("createAngle", &FixAngleCosineDelta::createAngle,
            (boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("theta0")=COEF_DEFAULT,
             boost::python::arg("type")=-1)
        )
    .def("setAngleTypeCoefs", &FixAngleCosineDelta::setAngleTypeCoefs,
            (boost::python::arg("type")=-1,
             boost::python::arg("k")=COEF_DEFAULT,
             boost::python::arg("theta0")=COEF_DEFAULT
            )
        )
    .def_readonly("angles", &FixAngleCosineDelta::pyForcers)
    ;
}

