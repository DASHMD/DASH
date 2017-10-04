#include "FixE3B3.h"

#include "BoundsGPU.h"
#include "GridGPU.h"
#include "State.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "list_macro.h"
#include "EvaluatorE3B3.h"
#include "ThreeBodyE3B3.h"

const std::string E3B3Type = "E3B3";
namespace py = boost::python;
/* Constructor
 * Makes an instance of the E3B3 fix
 */

FixE3B3::FixE3B3(boost::shared_ptr<State> state_,
                  std::string handle_,
                  std::string groupHandle_): Fix(state_, handle_, groupHandle_, E3B3Type, true, true, false, 1) { 
    // set the cutoffs used in this potential
    rf = 5.2; // far cutoff for threebody interactions (Angstroms)
    rs = 5.0; // short cutoff for threebody interactions (Angstroms)
    rc = 7.2; // cutoff for our local neighborlist (Angstroms)
    padding = 2.0; // implied since rc - rf = 2.0; pass this to local GridGPU on instantiation
    // to do: set up the local gridGPU for this set of GPUData; 
    // ---- which means we need to set up the local GPUData;
    // ------- can't do this until we have all the atoms in simulation; so do it in prepareForRun
};

//
// what arguments do we need here? we are updating the molecule positions from 
// the current atom positions

// from FixRigid.cu
__device__ inline float3 positionsToCOM_E3B3(float3 *pos, float *mass, float ims) {
  return (pos[0]*mass[0] + pos[1]*mass[1] + pos[2]*mass[2] + pos[3]*mass[3])*ims;
}

// useful for debugging
__global__ void printGPD_E3B3(uint* ids, float4 *xs, int nMolecules) {
    int idx = GETIDX();
    if (idx < nMolecules) {
        uint id = ids[idx];
        float4 pos = xs[idx];
        printf("molecule id %d at coords %f %f %f\n", id, pos.x, pos.y, pos.z);
    }
}

// prints the global gpd data
__global__ void printGPD_Global(uint* ids, float4 *xs, float4* vs, float4* fs, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint id = ids[idx];
        float4 pos = xs[idx];
        float4 vel = vs[idx];
        float4 force = fs[idx];
        printf("atom id %d at coords %f %f %f with vel %f %f %f and force %f %f %f\n", id, pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, force.x, force.y, force.z);
    }
}

// this function demonstrates how to properly access the neighborlist for E3B3,
// and prints out the results.  It should be found that all values are ~within sqrtf(neighCutSqr)
__global__ void printNlist_E3B3(int* molIdToIdxs,
                                uint* waterMolecIds,
                                int4* atomsFromMolecule,
                                uint16_t* neighborCounts,
                                uint* neighborlist,
                                uint32_t* cumulSumMaxPerBlock,
                                int warpSize,
                                int* idToIdxs,
                                float4* xs,
                                float4* fs, 
                                BoundsGPU bounds,
                                int nMolecules) {
    int idx = GETIDX();
    
    if (idx < nMolecules) {

        //int thisIdx = molIdToIdxs[waterMolecIds[idx]];

        // we need the following function:
        //inline __device__ int baseNeighlistIdxFromIndex(uint32_t *cumulSumMaxPerBlock, int warpSize, int idx) {

        int thisIdx = molIdToIdxs[waterMolecIds[idx]];
        printf("this Idx %d this id %d idx %d", thisIdx, waterMolecIds[idx], idx);
        //int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int baseIdx = baseNeighlistIdxFromIndex(cumulSumMaxPerBlock, warpSize, thisIdx);
        int numNeighMolecules = neighborCounts[thisIdx];
        //int numNeighMolecules = neighborCounts[idx];

        printf("idx %d baseIdx %d numNeighMolecules %d\n", thisIdx, baseIdx, numNeighMolecules);

        __syncthreads();
        // the neighborCounts and neighborList are for a given idx;
        //int4 atomsMolecule1 = atomsFromMolecule[idx];
        // atomsFromMolecule array is statically sorted by ids
        int4 atomsMolecule1 = atomsFromMolecule[waterMolecIds[idx]];
        int id_O1 = atomsMolecule1.x;
        int idx_a1 = idToIdxs[atomsMolecule1.x];
        int idx_b1 = idToIdxs[atomsMolecule1.y];
        int idx_c1 = idToIdxs[atomsMolecule1.z];

        float4 pos_a1_whole = xs[idx_a1];
        float4 pos_b1_whole = xs[idx_b1];
        float4 pos_c1_whole = xs[idx_c1];

        // now, get just positions in float3
        float3 pos_a1 = make_float3(pos_a1_whole);
        float3 pos_b1 = make_float3(pos_b1_whole);
        float3 pos_c1 = make_float3(pos_c1_whole);

        //int numNeigh = neighborCounts[idx];
        int numNeigh = neighborCounts[thisIdx];

        int counter = 0;
        for (int i = 0; i < numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize*i;
            uint otherIdxRaw = neighborlist[nlistIdx];

            int moleculeIds = waterMolecIds[otherIdxRaw];

            int4 atomsMolecule2 = atomsFromMolecule[moleculeIds];

            int id_O2 = atomsMolecule2.x;
            int id_B2 = atomsMolecule2.y;
            int id_C2 = atomsMolecule2.z;

            int idx_a2 = idToIdxs[atomsMolecule2.x];
            int idx_b2 = idToIdxs[atomsMolecule2.y];
            int idx_c2 = idToIdxs[atomsMolecule2.z];

            float4 pos_a2_whole = xs[idx_a2];
            float4 pos_b2_whole = xs[idx_b2];
            float4 pos_c2_whole = xs[idx_c2];
    
            // here we should extract the positions for the O, H atoms of this water molecule
            float3 pos_a2 = make_float3(pos_a2_whole);
            float3 pos_b2 = make_float3(pos_b2_whole);
            float3 pos_c2 = make_float3(pos_c2_whole);
            
            float3 r_b2a1 = bounds.minImage(pos_b2 - pos_a1);
            float3 r_c2a1 = bounds.minImage(pos_c2 - pos_a1);
            
            float3 r_b1a2 = bounds.minImage(pos_b1 - pos_a2);
            float3 r_c1a2 = bounds.minImage(pos_c1 - pos_a2);

            float r_b2a1_magnitude = length(r_b2a1);
            float r_c2a1_magnitude = length(r_c2a1);
            float r_b1a2_magnitude = length(r_b1a2);
            float r_c1a2_magnitude = length(r_c1a2);

            // we now have our molecule 'j'
            // compute the two-body correction term w.r.t the oxygens
            float3 r_a1a2 = bounds.minImage(pos_a1 - pos_a2);
            float r_a1a2_magnitude = length(r_a1a2);
            counter += 1;
            printf("O atom id %d neighbor ids %d %d %d a1-{a2,b2,c2} distances %f %f %f\n", id_O1, id_O2, id_B2, id_C2, r_a1a2_magnitude, r_b2a1_magnitude, r_c2a1_magnitude);

        }
        printf("finished printing out results for %d of %d neighbors\n", counter,numNeigh);
    }
}


// see FixRigid.cu! does the same thing. but now, we store it in their own gpdLocal..
// -- nothing fancy, dont need the neighborlist here.
__global__ void update_xs(int nMolecules, int4 *waterIds, float4 *mol_xs, int* mol_idToIdxs,
                           float4 *xs, float4 *vs, int *idToIdxs, BoundsGPU bounds) {

     // now do pretty much the same as FixRigid computeCOM()
     // --- remember to account for the M-site, in the event that it has mass
    int idx = GETIDX();
    
    if (idx < nMolecules) {

        // may as well make these arrays

        // just for clarity: we are looking at molecule /id/
        int molId = idx;

        int theseIds[4]; 
        float3 pos[4];
        float mass[4];

        // get the atom /ids/ for molecule id 'idx'
        theseIds[0] = waterIds[molId].x;
        theseIds[1] = waterIds[molId].y;
        theseIds[2] = waterIds[molId].z;
        theseIds[3] = waterIds[molId].w;

        float ims = 0.0f;
        // for each data (pos, vel, force), we need to get the position of atom id at position idToIdx in the global arrays
        for (int i = 0; i < 4; i++) {
            int thisId = theseIds[i];
            int thisIdx = idToIdxs[thisId];
            float3 p = make_float3(xs[thisIdx]);
            pos[i] = p;
            mass[i] = 1.0f / vs[thisIdx].w;
            ims += mass[i];
        }

        ims = 1.0f / ims;
        for (int i = 1; i < 4; i++) {
            float3 delta = pos[i] - pos[0];
            delta = bounds.minImage(delta);
            pos[i] = pos[0] + delta;
        }

        // and here is the COM of our water molecule
        mol_xs[mol_idToIdxs[molId]]  = make_float4(positionsToCOM_E3B3(pos, mass,ims));
        // and corresponding inverse mass
        mol_xs[mol_idToIdxs[molId]].w = ims;

    }

}

void FixE3B3::compute(int VirialMode) {
    
    // send the molecules to the e3b3 evaluator, where we compute both the two-body correction
    // and the three-body interactions.
    // -- send the correct neighbor list (specific to this potential) and the array of water molecules
    //    local to this gpu
    // -- still need to send the global simulation data, which contains the atoms itself
 

    bool computeVirials = false;
    // get the activeIdx for our local gpdLocal (the molecule-by-molecule stuff);
    int activeIdx = gpdLocal.activeIdx();

    // and the global gpd
    // --- IMPORTANT: the virials must be taken from the /global/ gpudata!
    GPUData &gpdGlobal = state->gpd;
    int globalActiveIdx = gpdGlobal.activeIdx();
    
    // our grid data holding our molecule-by-molecule neighbor list
    // -- we need to copy over the molecule array as well.
    
    // although it says 'perAtomArray', note that all of this gpd for this grid is by molecule
    // so, its just a misnomer in this instance. its a count of neighboring molecules.
    

    /* data required for compute_e3b3:
       - nMolecules
       - moleculesIdsToIdxs
       - waterIds (atom IDS in a given molecule)
       - molecules neighborcounts
       - molecules nlist
       - molecules - cumulSumMaxPerBlock (grid.perBlockArray.d_data.data())a
       - warpsize
       - atom idsToIdxs
       - atom positions
       - atom forces (....)
       - boundsGPU (state)
       - virials (global)
       - the evaluator
    */

    if (computeVirials) {
        compute_E3B3<EvaluatorE3B3, true> <<<NBLOCK(nMolecules), PERBLOCK>>> (
            nMolecules, 
            gpdLocal.idToIdxs.d_data.data(), 
            gpdLocal.ids(activeIdx),
            waterIdsGPU.data(),
            gridGPULocal.perAtomArray.d_data.data(),
            gridGPULocal.neighborlist.data(), 
            gridGPULocal.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize,
            gpdGlobal.idToIdxs.d_data.data(), 
            gpdGlobal.xs(globalActiveIdx), 
            gpdGlobal.fs(globalActiveIdx),
            state->boundsGPU, 
            gpdGlobal.virials.d_data.data(),
            evaluator);
    } else {
        compute_E3B3<EvaluatorE3B3, false> <<<NBLOCK(nMolecules), PERBLOCK>>> (
            nMolecules, 
            gpdLocal.idToIdxs.d_data.data(),
            gpdLocal.ids(activeIdx),
            waterIdsGPU.data(),
            gridGPULocal.perAtomArray.d_data.data(),
            gridGPULocal.neighborlist.data(), 
            gridGPULocal.perBlockArray.d_data.data(),
            state->devManager.prop.warpSize,
            gpdGlobal.idToIdxs.d_data.data(), 
            gpdGlobal.xs(globalActiveIdx), 
            gpdGlobal.fs(globalActiveIdx),
            state->boundsGPU, 
            gpdGlobal.virials.d_data.data(),
            evaluator);
    };
}


bool FixE3B3::stepInit(){
    // we use this as an opportunity to re-create the local neighbor list, if necessary
    int periodicInterval = state->periodicInterval;
    
    uint activeIdx = gpdLocal.activeIdx();

    // get the global gpd and the bounds
    uint globalActiveIdx = state->gpd.activeIdx();
    GPUData &gpdGlobal = state->gpd;
    BoundsGPU &bounds = state->boundsGPU;

    if (state->turn % periodicInterval == 0) {
        // do the re-creation of the neighborlist for E3B3
        // -- the xs of the molecules is /not/ updated with the atoms!
        //    but this is what we form our neighborlist off of (for the molecule-by-molecule neighborlist)
        //    so, do a kernel call here to update them to the current positions
        //    of their constituent atoms

        // update the positions of our molecules
        update_xs<<<NBLOCK(nMolecules), PERBLOCK>>>(nMolecules, 
                                                    waterIdsGPU.data(), 
                                                    gpdLocal.xs(activeIdx), 
                                                    gpdLocal.idToIdxs.d_data.data(),
                                                    gpdGlobal.xs(globalActiveIdx), 
                                                    gpdGlobal.vs(globalActiveIdx), 
                                                    gpdGlobal.idToIdxs.d_data.data(),
                                                    bounds
                                                    );
        // for each thread, we have one molecule
        // -- get the atoms for this idx, compute COM, set the xs to the new value, and return
        //    -- need idToIdx for atoms? I think so.  Also, this is easy place to check 
        //       accessing the data arrays

        // pass the local gpdLocal (molecule by molecule) and the global (atom by atom) gpd
        // -- -with this, our local gpdLocal data for the molecule COM is up to date with 
        //     the current atomic data
        cudaDeviceSynchronize();
    
        // our grid now operates on the updated molecule xs to get a molecule by molecule neighborlist    
        gridGPULocal.periodicBoundaryConditions();
        
    }

    return true;
}

/* Single Point Eng
 *
 *
 *
 */
//void FixE3B3::singlePointEng(float *perParticleEng) {
    // and, the three body contribution
    // -- we still pass everything molecule by molecule... but add it to their particle arrays

    // gonna need to look up how this is done..
    //return
//}



/* prepareForRun

   */
bool FixE3B3::prepareForRun(){
   
    // as angstroms
    float rs = 5.0;
    float rf = 5.2;

    // E2, Ea, Eb, Ec as kJ/mole -> convert to kcal/mole
    float E2 = 453000;
    float Ea = 150.0000;
    float Eb = -1005.0000;
    float Ec = 1880.0000;

    // k2, k3 as angstroms
    float k2 = 4.872;
    float k3 = 1.907;
    
    float kjToKcal = 0.23900573614;
    E2 *= kjToKcal;
    Ea *= kjToKcal;
    Eb *= kjToKcal;
    Ec *= kjToKcal;

    // 0 = REAL, 1 = LJ (see /src/Units.h)
    if (state->units.unitType == 1) {
        mdError("Units for E3B3 potential are not yet as LJ\n");
    }
        // converting to LJ from kcal/mol

    // instantiate the evaluator
    evaluator = EvaluatorE3B3(rs, rf, E2,
                              Ea, Eb, Ec,
                              k2, k3);
    
    nMolecules = waterMolecules.size();
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data()); // waterIds vector populated as molecs added
    
    std::vector<float4> xs_vec;
    std::vector<uint> ids;

    xs_vec.reserve(nMolecules);
    ids.reserve(nMolecules);

    
    int workingId = 0;
    for (auto &molecule: waterMolecules)  {
        molecule.id = workingId;
        Vector this_xs = molecule.COM();
        float4 new_xs = make_float4(this_xs[0], this_xs[1], this_xs[2], 0);
        xs_vec.push_back(new_xs);

        ids.push_back(molecule.id);
        workingId++;
    }

    // note that gpd is the /local/ gpd
    gpdLocal.xs.set(xs_vec);
    gpdLocal.ids.set(ids);
   
    std::vector<int> id_vec = LISTMAPREF(Molecule, int, m, waterMolecules, m.id);
    std::vector<int> idToIdxs_vec;
    int size = *std::max_element(id_vec.begin(), id_vec.end()) + 1;
    idToIdxs_vec.reserve(size);
    for (int i=0; i<size; i++) {
        idToIdxs_vec.push_back(-1);
    }
    for (int i=0; i<id_vec.size(); i++) {
        idToIdxs_vec[id_vec[i]] = i;
    }

    gpdLocal.idToIdxsOnCopy = idToIdxs_vec;
    gpdLocal.idToIdxs.set(idToIdxs_vec);
    gpdLocal.xs.dataToDevice();
    gpdLocal.ids.dataToDevice();
    gpdLocal.idToIdxs.dataToDevice();
    int activeIdx = gpdLocal.activeIdx();
    
    double maxRCut = rf;// cutoff of our potential (5.2 A)
    double padding = 1.0;
    double gridDim = maxRCut + padding;

    // this number has no meaning whatsoever; it is completely arbitrary;
    // -- we are not using exclusionMode for this grid or set of GPUData
    int exclusionMode = 30;
    // I think this is doubly irrelevant, since we use a doExclusions(false) method later (below)

    gridGPULocal = GridGPU(state, gridDim, gridDim, gridDim, gridDim, exclusionMode, padding, &gpdLocal);

    // tell gridGPU that the only GPUData we need to sort are positions (and, of course, the molecule/atom id's)
    gridGPULocal.onlyPositions(true);

    // tell gridGPU not to do any exclusions stuff
    gridGPULocal.doExclusions(false);

    // so, the only buffers that we need are the xs and ids!
    gpdLocal.xsBuffer = GPUArrayGlobal<float4>(nMolecules);
    gpdLocal.idsBuffer = GPUArrayGlobal<uint>(nMolecules);
    
    gridGPULocal.periodicBoundaryConditions(-1, true);

    prepared = true;
    return prepared;
}


/* restart chunk?


   */



/* postRun
   * nothing to do here

   */


// the atom ids are presented as the input; assembled into a molecule
void FixE3B3::addMolecule(int id_O, int id_H1, int id_H2, int id_M) {
    
    // id's are arranged as O, H, H, M
    std::vector<int> localWaterIds;

    // add to waterIds vector the four atom ids
    localWaterIds.push_back(id_O);
    localWaterIds.push_back(id_H1);
    localWaterIds.push_back(id_H2);
    localWaterIds.push_back(id_M);

    // mass of O > mass H1 == mass H2 > mass M
    bool ordered = true;
    double massO = state->idToAtom(id_O).mass; 
    double massH1 = state->idToAtom(id_H1).mass;
    double massH2 = state->idToAtom(id_H2).mass;
    double massM = state->idToAtom(id_M).mass;

    // check the ordering
    if (! (massO > massH1 && massO > massH2 )) {
        ordered = false;
    }
    if (massH1 != massH2) ordered = false;
    if (!(massH1 > massM)) ordered = false;

    if (! (ordered)) mdError("Ids in FixE3B3::addMolecule must be as O, H1, H2, M");

    // assemble them in to a molecule
    Molecule thisWater = Molecule(state, localWaterIds);

    // append this molecule to the class variable waterMolecules
    // -- molecule id is implicit as the index in this list
    waterMolecules.push_back(thisWater);

    int4 idsAsInt4 = make_int4(localWaterIds[0], localWaterIds[1], localWaterIds[2], localWaterIds[3]);
    // and add to the global list
    waterIds.push_back(idsAsInt4);


}

/* exports

   */

void export_FixE3B3() {
  py::class_<FixE3B3, boost::shared_ptr<FixE3B3>, py::bases<Fix> > 
	("FixE3B3",
         py::init<boost::shared_ptr<State>, std::string, std::string> 
	 (py::args("state", "handle", "groupHandle")
	 )
	)
    .def("addMolecule", &FixE3B3::addMolecule,
	     (py::arg("id_O"), 
          py::arg("id_H1"), 
          py::arg("id_H2"),
          py::arg("id_M")
         )
    );
}
