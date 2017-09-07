#include "FixSpringStatic.h"
#include "boost_for_export.h"
#include "FixHelpers.h"
#include "GPUData.h"
#include "State.h"
#include "xml_func.h"
namespace py = boost::python;

const std::string springStaticType = "SpringStatic";

FixSpringStatic::FixSpringStatic(boost::shared_ptr<State> state_,
                                 std::string handle_, std::string groupHandle_,
                                 double k_,  py::object tetherFunc_, Vector multiplier_)
  : Fix(state_, handle_, groupHandle_, springStaticType, true, false, false, 1),
    k(k_), tetherFunc(tetherFunc_), multiplier(multiplier_)
{
    updateTethers();
    readFromRestart();
    mdAssert(k!=-1, "spring k value not assigned");
}

void FixSpringStatic::updateTethers() {
    std::vector<float4> tethers_loc;
    PyObject *funcRaw = tetherFunc.ptr();
    if (PyCallable_Check(funcRaw)) {
        for (Atom &a : state->atoms) {
            if (a.groupTag & groupTag) {
                Vector res = boost::python::call<Vector>(funcRaw, a);
                tethers_loc.push_back(make_float4(res[0], res[1], res[2], *(float *)&a.id));
            }
        }
    } else {
        for (Atom &a : state->atoms) {
            if (a.groupTag & groupTag) {
                tethers_loc.push_back(make_float4(a.pos[0], a.pos[1], a.pos[2], *(float *)&a.id));
            }
        }
    }
    tethers = tethers_loc;
}


bool FixSpringStatic::prepareForRun() {
    tethers.dataToDevice();
    return true;
}

void __global__ compute_cu(int nTethers, float4 *tethers, float4 *xs, float4 *fs,
                           int *idToIdxs, float k,
                           BoundsGPU bounds, float3 multiplier) {
    int idx = GETIDX();
    if (idx < nTethers) {
        float4 tether = tethers[idx];
        float3 tetherPos = make_float3(tether);
        int id = * (int *) &tether.w;
        int atomIdx = idToIdxs[id];
        //printf("id for tether is %d idx is %d\n", id, idx);//, curPos.x, curPos.y, curPos.z);
        float3 curPos = make_float3(xs[atomIdx]);
        //printf("cur is %f %f, tether is %f %f, mult is %f %f %f, k is %f \n", curPos.x, curPos.y, tetherPos.x, tetherPos.y, multiplier.x, multiplier.y, multiplier.z, k);
        float3 force = multiplier * harmonicForce(bounds, curPos, tetherPos, k, 0);
        //printf("forces %f %f %f\n", force.x, force.y, force.z);
        fs[atomIdx] += force;
    }
}
void FixSpringStatic::compute(int virialMode) {
    GPUData &gpd = state->gpd;
    int activeIdx = state->gpd.activeIdx();
    compute_cu<<<NBLOCK(tethers.h_data.size()), PERBLOCK>>>(
                    tethers.h_data.size(), tethers.getDevData(),
                    gpd.xs(activeIdx), gpd.fs(activeIdx), gpd.idToIdxs.d_data.data(),
                    k, state->boundsGPU, multiplier.asFloat3());
}

std::string FixSpringStatic::restartChunk(std::string format) {
    std::stringstream ss;
    ss << "<multiplier>\n";
    for (int i=0; i<3; i++) {
        ss << multiplier[i] << "\n";
    }
    ss << "</multiplier>\n";
    ss << "<k>" << k << "</k>>\n";
    ss << "<tethers n=\"" << tethers.h_data.size() << "\">\n";
    for (float4 &tether : tethers.h_data) {
        ss << tether.x << " " << tether.y << " " << tether.z << " " << * (int *) &tether.w << "\n"; 
    }
    ss << "</tethers>\n";
    return ss.str();
}

bool FixSpringStatic::readFromRestart() {
    pugi::xml_node restData = getRestartNode();
    //params must be already initialized at this point (in constructor)
    if (restData) {

        auto curr_param = restData.first_child();
        while (curr_param) {
            curr_param = curr_param.next_sibling();
            std::string tag = curr_param.name();
            if (tag == "multiplier") {
                auto curr_pcnode = curr_param.first_child();
                int i=0;
                while (curr_pcnode) {
                    std::string data = curr_pcnode.value();
                    float x = atof(data.c_str());
                    multiplier[i] = x;
                    curr_pcnode = curr_pcnode.next_sibling();
                    i++;
                }

            } else if (tag == "k") {
                auto curr_pcnode = curr_param.first_child();
                std::string data = curr_pcnode.value();
                k = atof(data.c_str());
            } else if (tag == "tethers") {
                
                int n = boost::lexical_cast<int>(curr_param.attribute("n").value());
                std::vector<float4> tethers_loc(n);
                xml_assignValues<double, 4>(curr_param, [&] (int i, double *vals) { //doing as double to preserve precision for ids in w value
                                            int id = vals[3];
                                            float idAsFloat = *(float *) &id;
                                            tethers_loc[i] = make_float4(float(vals[0]), float(vals[1]), float(vals[2]), idAsFloat);
                                            });

                tethers = tethers_loc;

            }
        }

    }
    return true;

}

void export_FixSpringStatic() {
    py::class_<FixSpringStatic, boost::shared_ptr<FixSpringStatic>, py::bases<Fix> > (
            "FixSpringStatic",
            py::init<boost::shared_ptr<State>, std::string, std::string, double,
                     py::optional<py::object, Vector>
                    >(
                py::args("state", "handle", "groupHandle",
                         "k", "tetherFunc", "multiplier")
            )
    )
    .def("updateTethers", &FixSpringStatic::updateTethers)
    .def_readwrite("multiplier", &FixSpringStatic::multiplier)
    .def_readwrite("tetherFunc", &FixSpringStatic::tetherFunc)
    .def_readwrite("k", &FixSpringStatic::k)
    ;
}

