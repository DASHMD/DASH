#include "InitializeAtoms.h"
#include "State.h"
#include "Atom.h"
#include "list_macro.h"

#include "Logging.h"

using namespace std;

// make a 'ready' flag in state, which means am ready to run.  creating atoms
// makes false, make ready by re-doing all atom pointers
//
// nah, am making ready on each

int max_id(vector<Atom> &atoms) {
    int id = -1;
    for (Atom &a : atoms) {
        if (a.id > id) {
            id = a.id;
        }
    }
    return id;
}

void InitializeAtoms::populateOnGrid(SHARED(State) state, Bounds &bounds,
                                     string handle, int n) {
    assert(n>=0);
    vector<Atom> &atoms = state->atoms;

    int n_final = atoms.size() + n;

    int nPerSide = ceil(pow(n, 1.0/3.0));
    Vector deltaPerSide = bounds.rectComponents / nPerSide;
    for (int i=0; i<nPerSide; i++) {
        for (int j=0; j<nPerSide; j++) {
            for (int k=0; k<nPerSide; k++) {
                if ((int) atoms.size() == n_final) {
                    return;
                }
                Vector pos = bounds.lo + Vector(i, j, k) * deltaPerSide;
                state->addAtom(handle, pos, 0);
            }
        }
    }
}

void InitializeAtoms::populateRand(SHARED(State) state, Bounds &bounds,
                                   string handle, int n, double distMin) {
    assert(n>=0);

    std::mt19937 &generator = state->getRNG();
    vector<Atom> &atoms = state->atoms;
    AtomParams &params = state->atomParams;
    vector<string> handles = params.handles;
    int type = find(handles.begin(), handles.end(), handle) - handles.begin();

    assert(type != (int) handles.size()); //makes sure it found one
    unsigned int n_final = atoms.size() + n;
    uniform_real_distribution<double> dists[3];
    for (int i=0; i<3; i++) {
        dists[i] = uniform_real_distribution<double>(bounds.lo[i], bounds.lo[i] + bounds.rectComponents[i]);
    }
    if (state->is2d) {
        dists[2] = uniform_real_distribution<double>(0, 0);
    }

    int id = max_id(atoms) + 1;
    unsigned int tries = 0;
    while (atoms.size() < n_final) {
        Vector pos;
        for (int i=0; i<3; i++) {
            pos[i] = dists[i](generator);
        }
        bool is_overlap = false;
        for (Atom &a : atoms) {
            int typeA = a.type;
            /*! \todo Check only for overlap across boundary if boundary
             * is periodic. */
            Vector dist = state->bounds.minImage(pos - a.pos);
            if (dist.lenSqr() < distMin * distMin) {
                is_overlap = true;
                ++tries;
                if (tries > maxtries) { mdError("Unable to place new atom."); }
                break;
            }
        }
        if (not is_overlap) {
            state->addAtom(handle, pos, 0);
            id++;
            tries = 0;
        }
    }
    if (state->is2d) {
        for (Atom &a: atoms) {
            a.pos[2]=0;
        }
    }
}

void InitializeAtoms::initTemp(SHARED(State) state, string groupHandle,
                               double temp) {
    std::mt19937 generator = state->getRNG();
    int groupTag = state->groupTagFromHandle(groupHandle);

    vector<Atom *> atoms = LISTMAPREFTEST(Atom, Atom *, a, state->atoms, &a,
                                          a.groupTag & groupTag);

    assert(atoms.size());
    map<double, normal_distribution<double> > dists;
    for (Atom *a : atoms) {
        if (dists.find(a->mass) == dists.end()) {
            dists[a->mass] = normal_distribution<double>(0, sqrt(1.0/a->mass));
        }
    }
    Vector sumMoms;
    double sumMass = 0;
    for (Atom *a : atoms) {
        for (int i=0; i<3; i++) {
            a->vel[i] = dists[a->mass](generator);
        }
        sumMoms += a->vel * a->mass;
        sumMass += a->mass;
    }
    if (atoms.size()>1) {
        sumMoms /= sumMass;
        for (Atom *a : atoms) {
            a->vel -= sumMoms;
        }
    }
    double sumKe = 0;
    for (Atom *a : atoms) {
        if (state->is2d) {
            a->vel[2] = 0;
        }
        sumKe += 2.0 * a->kinetic();
    }
    double curTemp = (state->units.mvv_to_eng / state->units.boltz) * sumKe / ((state->is2d ? 2 : 3) * (atoms.size()));
    for (Atom *a : atoms) {
        a->vel *= sqrt(temp / curTemp);
    }
    sumKe = 0;
    for (Atom *a : atoms) {
        if (state->is2d) {
            a->vel[2] = 0;
        }
        sumKe += 2.0 * a->kinetic();
    }

}

void export_InitializeAtoms() {
    boost::python::class_<InitializeAtomsPythonWrap> (
        "InitializeAtoms"
    )
    //.def("populateOnGrid", &InitializeAtoms::populateOnGrid,
    //        (boost::python::arg("bounds"),
    //         boost::python::arg("handle"),
    //         boost::python::arg("n"))
    //    )
    //.staticmethod("populateOnGrid")
    .def("populateRand", &InitializeAtoms::populateRand,
            (boost::python::arg("bounds"),
             boost::python::arg("handle"),
             boost::python::arg("n"),
             boost::python::arg("distMin"))
        )
    .staticmethod("populateRand")
    .def("initTemp", &InitializeAtoms::initTemp,
            (boost::python::arg("groupHandle"),
             boost::python::arg("temp"))
        )
    .staticmethod("initTemp")
    ;
}

