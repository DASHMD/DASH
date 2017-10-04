#include "FixRigid.h"

#include "State.h"
#include "Atom.h"
#include "VariantPyListInterface.h"
#include "boost_for_export.h"
#include "cutils_math.h"
#include "cutils_func.h"
#include <math.h>
#include "globalDefs.h"
#include "xml_func.h"
#include "../Eigen/Dense"
namespace py = boost::python;
const std::string rigidType = "Rigid";

FixRigid::FixRigid(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_) : Fix(state_, handle_, groupHandle_, rigidType, true, true, false, 1) {

    // set both to false initially; using one of the createRigid functions will flip the pertinent flag to true
    TIP4P = false;
    TIP3P = false;
    printing = false;
    // set flag 'requiresPostNVE_V' to true
    requiresPostNVE_V = true;
    //requiresPostNVE_V = true;
    style = "DEFAULT";
    // this fix requires the forces to have already been computed before we can 
    // call prepareForRun()
    requiresForces = true;
    solveInitialConstraints = true;
    readFromRestart();

}


inline __host__ __device__ float3 rotation(float3 vector, float3 X, float3 Y, float3 Z) {
    return make_float3(dot(X,vector), dot(Y,vector), dot(Z, vector));
}

inline __host__ __device__ double3 rotation(double3 vector, double3 X, double3 Y, double3 Z) {
    return make_double3(dot(X,vector), dot(Y,vector), dot(Z, vector));
}

// verified to be correct
inline __host__ __device__ double matrixDet(double3 ROW1, double3 ROW2, double3 ROW3) 
{
    // [a   b    c]
    // [d   e    f]  = A
    // [g   h    i]
    //
    // det(A) = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    //
    //
    double det;
    // so, get the first value..
    //         a   * (  e    *   i    -   f    *   h   )
    det =   ROW1.x * (ROW2.y * ROW3.z - ROW2.z * ROW3.y);
    //  -      b   * (  d    *   i    -   f    *   g   )
    det -= (ROW1.y * (ROW2.x * ROW3.z - ROW2.z * ROW3.x));
    //  +      c   * (  d    *   h    -   e    *   g   )
    det += (ROW1.z * (ROW2.x * ROW3.y - ROW2.y * ROW3.x));
    return det;
}


// kind of just need this for validation that M*M^-1 results in the identity
// ---- and it does! May as well keep it, though.
// verified to be correct
inline __host__ __device__ void matrixMultiplication(double3 A1, double3 A2, double3 A3,
                                                     double3 B1, double3 B2, double3 B3,
                                                     double3 &C1, double3 &C2, double3 &C3) {


    C1.x = A1.x * B1.x + A1.y * B2.x + A1.z * B3.x;
    C2.x = A2.x * B1.x + A2.y * B2.x + A2.z * B3.x;
    C3.x = A3.x * B1.x + A3.y * B2.x + A3.z * B3.x;

    C1.y = A1.x * B1.y + A1.y * B2.y + A1.z * B3.y;
    C2.y = A2.x * B1.y + A2.y * B2.y + A2.z * B3.y;
    C3.y = A3.x * B1.y + A3.y * B2.y + A3.z * B3.y;

    C1.z = A1.x * B1.z + A1.y * B2.z + A1.z * B3.z;
    C2.z = A2.x * B1.z + A2.y * B2.z + A2.z * B3.z;
    C3.z = A3.x * B1.z + A3.y * B2.z + A3.z * B3.z;

}

inline __host__ __device__ double3 matrixVectorMultiply(double3 A1, double3 A2, double3 A3, double3 V)
{
    return make_double3( A1.x * V.x + A1.y * V.y + A1.z * V.z,
                         A2.x * V.x + A2.y * V.y + A2.z * V.z,
                         A3.x * V.x + A3.y * V.y + A3.z * V.z);


}

// verified to be correct
inline __host__ __device__ void invertMatrix(double3 ROW1, double3 ROW2, double3 ROW3, double3 &invROW1, double3 &invROW2, double3 &invROW3)
{
    // get the inverse determinant, and then compute the individual elements of the rows as required for the 
    // inverse 'matrix': three rows of double3's
    double det = matrixDet(ROW1, ROW2, ROW3);
    double invDet = 1.0 / det;

    invROW1.x =        invDet * (ROW2.y * ROW3.z - ROW3.y * ROW2.z);
    invROW1.y = -1.0 * invDet * (ROW1.y * ROW3.z - ROW3.y * ROW1.z);
    invROW1.z =        invDet * (ROW1.y * ROW2.z - ROW2.y * ROW1.z);

    invROW2.x = -1.0 * invDet * (ROW2.x * ROW3.z - ROW3.x * ROW2.z);
    invROW2.y =        invDet * (ROW1.x * ROW3.z - ROW3.x * ROW1.z);
    invROW2.z = -1.0 * invDet * (ROW1.x * ROW2.z - ROW2.x * ROW1.z);
    
    invROW3.x =        invDet * (ROW2.x * ROW3.y - ROW3.x * ROW2.y);
    invROW3.y = -1.0 * invDet * (ROW1.x * ROW3.y - ROW3.x * ROW1.y);
    invROW3.z =        invDet * (ROW1.x * ROW2.y - ROW2.x * ROW1.y);

    // and done

}



// so this is the exact same function as in FixLinearMomentum. 
__global__ void rigid_remove_COMV(int nAtoms, float4 *vs, float4 *sumData, float3 dims) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 v = vs[idx];
        float4 sum = sumData[0];
        float invMassTotal = 1.0f / sum.w;
        v.x -= sum.x * invMassTotal * dims.x;
        v.y -= sum.y * invMassTotal * dims.y;
        v.z -= sum.z * invMassTotal * dims.z;
        vs[idx] = v;
    }
}

template <class DATA, bool TIP4P>
__global__ void rigid_scaleSystem_cu(int4* waterIds, float4* xs, int* idToIdxs,
                                     float3 lo, float3 rectLen, BoundsGPU bounds, float3 scaleBy,
                                     DATA fixRigidData, int nMolecules) {

    int idx = GETIDX();
    if (idx < nMolecules) {
        // compute the COM; 
        // perform the displacement;
        // compute the difference in the positions
        // translate all the individual atoms xs's accordingly, and exit
        // --- we do not need to check groupTags
        // --- if (TIP4P) { translate M-site position as well }
        double weightH = fixRigidData.weights.y;
       
        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole positions
        float4 posO_whole = xs[idxO];
        float4 posH1_whole= xs[idxH1];
        float4 posH2_whole= xs[idxH2];

        // and the unconstrained, updated positions
        double3 posO = make_double3(posO_whole);
        double3 posH1= make_double3(posH1_whole);
        double3 posH2= make_double3(posH2_whole);
    
        // get the relative vectors for the unconstrained triangle
        double3 OH1_unconstrained = bounds.minImage(posH1 - posO);
        double3 OH2_unconstrained = bounds.minImage(posH2 - posO);

        // move the hydrogens to their minimum image distance w.r.t. the oxygen
        posH1 = posO + OH1_unconstrained;
        posH2 = posO + OH2_unconstrained;

        // the center of mass 'd1' in the paper
        double3 COM_d1 = (posO ) - ( ( posH1 + posH2 ) * weightH);
    
        // do as Mod::scale on COM_d1
        double3 center = make_double3(lo) + make_double3(rectLen) * 0.5;

        double3 newRel = (COM_d1 - center) * make_double3(scaleBy);

        // this is the vector which we will add to the positions of our atoms
        double3 diff = center - COM_d1;

        posO += diff;
        posH1 += diff;
        posH2 += diff;
        if (TIP4P) {
            int idxM = idToIdxs[atomsFromMolecule.w];
            float4 posM_whole = xs[idxM];
            double3 posM = make_double3(posM_whole);
            posM += diff;
            float3 newPosM = make_float3(posM);
            xs[idxM] = make_float4(newPosM, posM_whole.w);
        }

        float3 newPosO = make_float3(posO);
        float3 newPosH1= make_float3(posH1);
        float3 newPosH2= make_float3(posH2);

        xs[idxO] = make_float4(newPosO, posO_whole.w);
        xs[idxH1]= make_float4(newPosH1,posH1_whole.w);
        xs[idxH2]= make_float4(newPosH2,posH2_whole.w);
    }
}



template <class DATA, bool TIP4P>
__global__ void rigid_scaleSystemGroup_cu(int4* waterIds, float4* xs, int* idToIdxs,
                                          float3 lo, float3 rectLen, BoundsGPU bounds, float3 scaleBy,
                                          DATA fixRigidData, int nMolecules, uint32_t groupTag,
                                          float4* fs) {

    int idx = GETIDX();
    if (idx < nMolecules) {
        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];
        int idxO = idToIdxs[atomsFromMolecule.x];
        uint32_t tag = * (uint32_t *) &(fs[idxO].w);
        if (tag & groupTag) {
            // compute the COM; 
            // perform the displacement;
            // compute the difference in the positions
            // translate all the individual atoms xs's accordingly, and exit
            // ---- check the groupTag of just the oxygen; if 
            //      the oxygen atom is in the group, we will assume that the other atoms are as well
            //      mostly because if they aren't, then something is being done incorrectly by the user
            // --- if (TIP4P) { translate M-site position as well }
            double weightO = fixRigidData.weights.x;
            double weightH = fixRigidData.weights.y;
           

            // get the atom idxs
            int idxH1= idToIdxs[atomsFromMolecule.y];
            int idxH2= idToIdxs[atomsFromMolecule.z];
            
            // extract the whole positions
            float4 posO_whole = xs[idxO];
            float4 posH1_whole= xs[idxH1];
            float4 posH2_whole= xs[idxH2];

            // and the unconstrained, updated positions
            double3 posO = make_double3(posO_whole);
            double3 posH1= make_double3(posH1_whole);
            double3 posH2= make_double3(posH2_whole);
        
            // get the relative vectors for the unconstrained triangle
            double3 OH1_unconstrained = bounds.minImage(posH1 - posO);
            double3 OH2_unconstrained = bounds.minImage(posH2 - posO);

            // move the hydrogens to their minimum image distance w.r.t. the oxygen
            posH1 = posO + OH1_unconstrained;
            posH2 = posO + OH2_unconstrained;

            // the center of mass 'd1' in the paper
            double3 COM_d1 = (posO * weightO) + ( ( posH1 + posH2 ) * weightH);
        
            // do as Mod::scale on COM_d1
            double3 center = make_double3(lo) + make_double3(rectLen) * 0.5;

            double3 newRel = (COM_d1 - center) * make_double3(scaleBy);

            // this is the vector which we will add to the positions of our atoms
            double3 diff = center - COM_d1;

            posO += diff;
            posH1 += diff;
            posH2 += diff;
            if (TIP4P) {
                int idxM = idToIdxs[atomsFromMolecule.w];
                float4 posM_whole = xs[idxM];
                double3 posM = make_double3(posM_whole);
                posM += diff;
                float3 newPosM = make_float3(posM);
                xs[idxM] = make_float4(newPosM, posM_whole.w);
            }

            float3 newPosO = make_float3(posO);
            float3 newPosH1= make_float3(posH1);
            float3 newPosH2= make_float3(posH2);

            xs[idxO] = make_float4(newPosO, posO_whole.w);
            xs[idxH1]= make_float4(newPosH1,posH1_whole.w);
            xs[idxH2]= make_float4(newPosH2,posH2_whole.w);

        }


    }

}

// called at the end of stepFinal, this will be silent unless something is amiss (if the constraints are not satisifed for every molecule)
// i.e., bond lengths should be fixed, and the dot product of the relative velocities along a bond with the 
// bond vector should be identically zero.
template <class DATA>
__global__ void validateConstraints(int4* waterIds, int *idToIdxs, 
                                    float4 *xs, float4 *vs, 
                                    int nMolecules, BoundsGPU bounds, 
                                    DATA fixRigidData, bool* constraints, int turn) {

    int idx = GETIDX();

    if (idx < nMolecules) {

        // extract the ids
        int id_O =  waterIds[idx].x;
        int id_H1 = waterIds[idx].y;
        int id_H2 = waterIds[idx].z;

        // so these are the actual OH, OH, HH, OM bond lengths, not the sides of the canonical triangle
        // save them as double4, cast them as float for arithmetic, since really all calculations are done in float
        double4 sideLengths = fixRigidData.sideLengths;

        // (re)set the constraint boolean
        constraints[idx] = true;

        // get the positions
        float3 pos_O = make_float3(xs[idToIdxs[id_O]]);
        float3 pos_H1 = make_float3(xs[idToIdxs[id_H1]]);
        float3 pos_H2 = make_float3(xs[idToIdxs[id_H2]]);

        // get the velocities
        float3 vel_O = make_float3(vs[idToIdxs[id_O]]);
        float3 vel_H1= make_float3(vs[idToIdxs[id_H1]]);
        float3 vel_H2= make_float3(vs[idToIdxs[id_H2]]);

        // our constraints are that the 
        // --- OH1, OH2, H1H2 bond lengths are the specified values;
        // --- the dot product of the bond vector with the relative velocity along the bond is zero
        
        // take a look at the bond lengths first
        // i ~ O, j ~ H1, k ~ H2
        float3 r_ij = bounds.minImage(pos_H1 - pos_O);
        float3 r_ik = bounds.minImage(pos_H2 - pos_O);
        float3 r_jk = bounds.minImage(pos_H2 - pos_H1);

        float len_rij = length(r_ij);
        float len_rik = length(r_ik);
        float len_rjk = length(r_jk);
    
        // side length AB (AB = AC) (AB ~ intramolecular OH bond)

        float AB = (float) sideLengths.x;
        float BC = (float) sideLengths.z;

        // and these should be ~0.0f
        float bondLenij = len_rij - AB;
        float bondLenik = len_rik - AB;
        float bondLenjk = len_rjk - BC;

        // these values correspond to the fixed side lengths of the triangle congruent to the specific water model being examined

        // now take a look at dot product of relative velocities along the bond vector
        float3 v_ij = vel_H1 - vel_O;
        float3 v_ik = vel_H2 - vel_O;
        float3 v_jk = vel_H2 - vel_H1;

        float mag_v_ij = length(v_ij);
        float mag_v_ik = length(v_ik);
        float mag_v_jk = length(v_jk);

        // this is the "relative velocity along the bond" constraint
        float bond_ij = dot(r_ij, v_ij);
        float bond_ik = dot(r_ik, v_ik);
        float bond_jk = dot(r_jk, v_jk);
        /*
        float constr_relVel_ij = bond_ij / mag_v_ij;
        float constr_relVel_ik = bond_ik / mag_v_ik;
        float constr_relVel_jk = bond_jk / mag_v_jk;
        */
        // 1e-3;
        // so, we only ever have ~7 digits of precision (log10(2^24) = 7.22....) due to the global variables
        float tolerance = 0.001;
        // note that these values should all be zero
        if ( ( bond_ij > tolerance) or 
             ( bond_ik > tolerance) or
             ( bond_jk > tolerance) ) {
            constraints[idx] = false;
            printf("water molecule %d unsatisfied velocity constraints at turn %d,\ndot(r_ij, v_ij) for ij = {01, 02, 12} %f, %f, and %f; tolerance %f\n", idx, (int) turn,
                    bond_ij, bond_ik, bond_jk, tolerance);
        }
        
        if ( (fabs(bondLenij) > tolerance) or
             (fabs(bondLenik) > tolerance) or 
             (fabs(bondLenjk) > tolerance)) {
            // then the position (bond length) constraints are not satisfied
            constraints[idx] = false;
            printf("water molecule %d did not have position constraints satisfied at turn %d\nExpected bond lengths (OH1, OH2, H1H2) of %f %f %f, got %f %f %f; tolerance is currently %f\n", idx, (int) turn,
                   AB, AB, BC, len_rij, len_rik, len_rjk, tolerance);
        }
    }
}


// pretty straightforward - printing out the positions, velocities, and forces of 4 atoms                                   
__global__ void printGPD_Rigid(uint* ids, float4 *xs, float4 *vs, float4 *fs, int nAtoms) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        uint id = ids[idx];
        //if (id < 4) {
        float4 pos = xs[idx];
        int type = xs[idx].w;
        float4 vel = vs[idx];
        float4 force = fs[idx];
        //uint groupTag = force.w;
        printf("atom id %d type %d at coords %f %f %f\n", id, type, pos.x, pos.y, pos.z);
        printf("atom id %d mass %f with vel  %f %f %f\n", id, vel.w, vel.x, vel.y, vel.z);
        printf("atom id %d mass %f with force %f %f %f\n", id, vel.w, force.x, force.y, force.z);
        //}
    }
}


// distribute the m site forces, and do an unconstrained integration of the velocity component corresponding to this additional force
// -- this is required for 4-site models with a massless site.
template <bool VIRIALS,bool FORCES>
__global__ void distributeMSite(int4 *waterIds, float4 *xs, float4 *vs, float4 *fs, 
                                Virial *virials,
                                int nMolecules, float gamma, float dtf, int* idToIdxs, BoundsGPU bounds)

{
    int idx = GETIDX();
    if (idx < nMolecules) {
        // by construction, the id's of the molecules are ordered as follows in waterIds array

        // waterIdxs contains id's; we need idxs
        int idx_O  = idToIdxs[waterIds[idx].x];
        int idx_H1 = idToIdxs[waterIds[idx].y];
        int idx_H2 = idToIdxs[waterIds[idx].z];
        int idx_M  = idToIdxs[waterIds[idx].w];

        if (FORCES) {
            float4 vel_O = vs[idx_O];
            float4 vel_H1 = vs[idx_H1];
            float4 vel_H2 = vs[idx_H2];

            // need the forces from O, H1, H2, and M
            float4 fs_O  = fs[idx_O];
            float4 fs_H1 = fs[idx_H1];
            float4 fs_H2 = fs[idx_H2];
            float4 fs_M  = fs[idx_M];

            // -- these are the forces from the M-site partitioned for distribution to the atoms of the water molecule
            float3 fs_O_d = make_float3(fs_M) * gamma;
            float3 fs_H_d = make_float3(fs_M) * 0.5 * (1.0 - gamma);

            // get the inverse masses from velocity variables above
            float invMassO = vel_O.w;

            // if the hydrogens don't have equivalent masses, we have bigger problems
            float invMassH = vel_H1.w;

            // compute the differential addition to the velocities
            float3 dv_O = dtf * invMassO * fs_O_d;
            float3 dv_H = dtf * invMassH * fs_H_d;

            // and add to the velocities of the atoms
            vel_O  += dv_O;
            vel_H1 += dv_H;
            vel_H2 += dv_H;

            // set the velocities to the new velocities in vel_O, vel_H1, vel_H2
            vs[idx_O] = vel_O; 
            vs[idx_H1]= vel_H1;
            vs[idx_H2]= vel_H2;
           
            vs[idx_M] = make_float4(0,0,0,INVMASSLESS);
            // finally, modify the forces; this way, the distributed force from M-site is incorporated in to nve_v() integration step
            // at beginning of next iteration in IntegratorVerlet.cu
            fs_O += fs_O_d;
            fs_H1 += fs_H_d;
            fs_H2 += fs_H_d;
           
            // set the global variables *fs[idToIdx[id]] to the new values
            fs[idx_O] = fs_O;
            fs[idx_H1]= fs_H1;
            fs[idx_H2]= fs_H2;

            // zero the force on the M-site, just because
            fs[idx_M] = make_float4(0.0, 0.0, 0.0,fs_M.w);
        } 


        if (VIRIALS) {
            Virial virialToDistribute = virials[idx_M];
            
            Virial distribute_O = virialToDistribute * (1.0 - (2.0 * gamma));
            Virial distribute_H = virialToDistribute * gamma;
            
            virials[idx_O]  += distribute_O;
            virials[idx_H1] += distribute_H;
            virials[idx_H2] += distribute_H;
            virials[idx_M] = Virial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        // this concludes re-distribution of the forces;
        // we assume nothing needs to be done re: virials; this sum is already tabulated at inner force loop computation
        // in the evaluators; for safety, we might just set 

    }
}

template <class DATA>
__global__ void setMSite(int4 *waterIds, int *idToIdxs, float4 *xs, int nMolecules, BoundsGPU bounds, DATA fixRigidData) {

    int idx = GETIDX();
    if (idx < nMolecules) {
    
        /* What we do here:
         * get the minimum image positions of the O, H, H atoms
         * compute the vector position of the M site
         * apply PBC to this new position (in case the water happens to be on the boundary of the box
         */

        // first, get the ids of the atoms composing this molecule
        int idx_O  = idToIdxs[waterIds[idx].x];
        int idx_H1 = idToIdxs[waterIds[idx].y];
        int idx_H2 = idToIdxs[waterIds[idx].z];
        int idx_M  = idToIdxs[waterIds[idx].w];

        float4 pos_M_whole = xs[idx_M];
        
        // get the positions of said atoms

        float3 pos_O = make_float3(xs[idx_O]);
        float3 pos_H1= make_float3(xs[idx_H1]);
        float3 pos_H2= make_float3(xs[idx_H2]);
        float3 pos_M = make_float3(xs[idx_M]);

        // compute vectors r_ij and r_ik according to minimum image convention
        // where r_ij = r_j - r_i, r_ik = r_k - r_i,
        // and r_i, r_j, r_k are the 3-component vectors describing the positions of O, H1, H2, respectively
        float3 r_ij = bounds.minImage( (pos_H1 - pos_O));
        float3 r_ik = bounds.minImage( (pos_H2 - pos_O));

        // fixRigidData.sideLengths.w is the OM vector
        float3 r_M  = (pos_O) + fixRigidData.sideLengths.w * ( (r_ij + r_ik) / ( (length(r_ij + r_ik))));
    
        float4 pos_M_new = make_float4(r_M.x, r_M.y, r_M.z, pos_M_whole.w);
        xs[idx_M] = pos_M_new;
    }
}


// computes the center of mass for a given water molecule
template <class DATA>
__global__ void compute_COM(int4 *waterIds, float4 *xs, float4 *vs, int *idToIdxs, int nMolecules, float4 *com, BoundsGPU bounds, DATA fixRigidData) {
  int idx = GETIDX();
  if (idx  < nMolecules) {
    float3 pos[3];

    double4 mass = fixRigidData.weights;
    int ids[3];
    ids[0] = waterIds[idx].x;
    ids[1] = waterIds[idx].y;
    ids[2] = waterIds[idx].z;
    for (int i = 0; i < 3; i++) {
      int myId = ids[i];
      int myIdx = idToIdxs[myId];
      float3 p = make_float3(xs[myIdx]);
      pos[i] = p;
      }
    for (int i=1; i<3; i++) {
      float3 delta = pos[i] - pos[0];
      delta = bounds.minImage(delta);
      pos[i] = pos[0] + delta;
    }
    float ims = com[idx].w;
    com[idx] = make_float4((pos[0] * mass.x) + ( (pos[1] + pos[2]) * mass.y));
    com[idx].w = ims;
  }
}

// save the previous positions of the constraints to use as the basis for our next constraint solution
// --- note that we do not care about the M-site in this instance;
__global__ void save_prev_val(int4 *waterIds, float4 *xs, float4 *xs_0, int nMolecules, int *idToIdxs) {
  int idx = GETIDX();
  if (idx < nMolecules) {
    int ids[3];

    // get the atom id's associated with this molecule idx

    // id of Oxygen
    ids[0] = waterIds[idx].x;
    // id of H1
    ids[1] = waterIds[idx].y;
    // id of H2
    ids[2] = waterIds[idx].z;
    // for O, H1, H2:
    for (int i = 0; i < 3; i++) {
      // get the idx of this atom id (O, H1, H2)
      int myIdx = idToIdxs[ids[i]];
      // store the xs initial data 
      xs_0[idx*3 + i] = xs[myIdx];
    }
  }
}


template <class DATA, bool VIRIALS, bool DEBUG_BOOL>
__global__ void settleVelocities(int4 *waterIds, float4 *xs, float4 *xs_0, 
                               float4 *vs, double3 *velCorrectionStored,
                               float4 *fs, Virial *virials, float4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               float dt, float dtf,
                               int *idToIdxs, BoundsGPU bounds, int turn) {
    int idx = GETIDX();
    if (idx < nMolecules) {
        

        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole velocities
        float4 velO_whole = vs[idxO];
        float4 velH1_whole= vs[idxH1];
        float4 velH2_whole= vs[idxH2];

        // convert to double3 (drop the inv mass)
        double3 velO = make_double3(velO_whole);
        double3 velH1= make_double3(velH1_whole);
        double3 velH2= make_double3(velH2_whole);


        double3 O_corr, H1_corr, H2_corr;
        float3 velO_tmp, velH1_tmp, velH2_tmp;

        // and our positions - get the current sidelengths
        float4 posO_whole  = xs[idxO];
        float4 posH1_whole = xs[idxH1];
        float4 posH2_whole = xs[idxH2];
        
        // and cast as double
        double3 xO = make_double3(posO_whole);
        double3 xH1= make_double3(posH1_whole);
        double3 xH2= make_double3(posH2_whole);
     
        double imO = fixRigidData.invMasses.z;
        double imH = fixRigidData.invMasses.w;
        double dOH = fixRigidData.sideLengths.x;
        double dHH = fixRigidData.sideLengths.z;
        double invdOH = fixRigidData.invSideLengths.x;
        double invdHH = fixRigidData.invSideLengths.z;
       

        if (DEBUG_BOOL) {

            xO   = make_double3(17.4575408690381, 28.0674379569203, 19.7605008286979);
            xH1  = make_double3(18.3859982152049, 28.1295117460303, 19.5361224851473);
            xH2  = make_double3(17.3030341458184, 27.1311540794149, 19.8859281510936);

            // ow1 x,y,z pos:   1.74575408690381     2.80674379569203     1.97605008286979
            // hw2 x,y,z pos:   1.83859982152049     2.81295117460303     1.95361224851473
            // hw3 x,y,z pos:   1.73030341458184     2.71311540794149     1.98859281510936

            velO = make_double3(-0.12463504629767, -0.11015356806752, 0.02384241975976)*0.01;
            velH1= make_double3(-0.29861852567439, 2.32000214713289, -0.15679017362187)*0.01;
            velH2= make_double3(2.27687840636365, -0.57159838032778, -0.22164674175936)*0.01;


            //ow1 x,y,z der:  -0.12463504629767    -0.11015356806752     0.02384241975976
            //hw2 x,y,z der:  -0.29861852567439     2.32000214713289    -0.15679017362187
            //hw3 x,y,z der:   2.27687840636365    -0.57159838032778    -0.22164674175936
        

            double3 invmat_0 = fixRigidData.M1_inv;
            double3 invmat_1 = fixRigidData.M2_inv;
            double3 invmat_2 = fixRigidData.M3_inv;
            printf("\nIn settleProj routine!\nimO: %18.14f\nimH: %18.14f\ndOH: %18.14f\ndHH: %18.14f\ninvdOH: %18.14f\ninvdHH: %18.14f\n",
                    imO, imH, dOH, dHH, invdOH, invdHH);
            printf("invmat[0][0]: %18.14f\ninvmat[0][1]: %18.14f\ninvmat[0][2]: %18.14f\ninvmat[1][0]: %18.14f\ninvmat[1][1]: %18.14f\ninvmat[1][2]: %18.14f\ninvmat[2][0]: %18.14f\ninvmat[2][1]: %18.14f\ninvmat[2][2]: %18.14f\n",
                   invmat_0.x, invmat_0.y, invmat_0.z,
                   invmat_1.x, invmat_1.y, invmat_1.z,
                   invmat_2.x, invmat_2.y, invmat_2.z);
            
            printf("ow1 x,y,z, pos: %18.14f  %18.14f   %18.14f\n",
                   xO.x, xO.y, xO.z);
            printf("hw2 x,y,z, pos: %18.14f  %18.14f   %18.14f\n",
                   xH1.x, xH1.y, xH1.z);
            printf("hw3 x,y,z, pos: %18.14f  %18.14f   %18.14f\n",
                   xH2.x, xH2.y, xH2.z);
            
            printf("ow1 x,y,z der: %18.14f   %18.14f   %18.14f\n",
                   velO.x, velO.y, velO.z);
            printf("hw2 x,y,z der: %18.14f   %18.14f   %18.14f\n",
                   velH1.x, velH1.y, velH1.z);
            printf("hw3 x,y,z der: %18.14f   %18.14f   %18.14f\n",
                   velH2.x, velH2.y, velH2.z);
            printf("derp before modification\n");

            printf("ow1 x,y,z derp: %18.14f   %18.14f   %18.14f\n",
                   velO.x, velO.y, velO.z);
            printf("hw2 x,y,z derp: %18.14f   %18.14f   %18.14f\n",
                   velH1.x, velH1.y, velH1.z);
            printf("hw3 x,y,z derp: %18.14f   %18.14f   %18.14f\n",
                   velH2.x, velH2.y, velH2.z);

        }



        double3 rOH1 = bounds.minImage(xO-xH1);
        double3 rOH2 = bounds.minImage(xO-xH2);
        double3 rH1H2  = bounds.minImage(xH1-xH2);

        // inverse bond lengths as stipulated by the fixed geometry
        double inverseROH = fixRigidData.invSideLengths.x;
        double inverseRHH = fixRigidData.invSideLengths.z;


        if (DEBUG_BOOL) {
            printf("rOH1 vals: %18.14f  %18.14f  %18.14f\n",
                    rOH1.x, rOH1.y, rOH1.z);
            printf("rOH2 vals: %18.14f  %18.14f  %18.14f\n",
                    rOH2.x, rOH2.y, rOH2.z);
            printf("rHH  vals: %18.14f  %18.14f  %18.14f\n",
                    rH1H2.x, rH1H2.y, rH1H2.z);
        }
        // these are the vectors along which the forces are applied 
        // --- keep this in mind for the Virials!
        rOH1   *= inverseROH;
        rOH2   *= inverseROH;
        rH1H2  *= inverseRHH;

        if (DEBUG_BOOL) {
            printf("calling svmul on roh2, roh3, rhh..\n");
            printf("rOH1 vals: %18.14f  %18.14f  %18.14f\n",
                    rOH1.x, rOH1.y, rOH1.z);
            printf("rOH2 vals: %18.14f  %18.14f  %18.14f\n",
                    rOH2.x, rOH2.y, rOH2.z);
            printf("rHH  vals: %18.14f  %18.14f  %18.14f\n",
                    rH1H2.x, rH1H2.y, rH1H2.z);
        }
        // OKAY, so, everything up to here is confirmed correct..
        double3 relativeVelocity;

        // set the x, y, z components as required.  Keep orientation O<-->H consistent with the force projection 
        // matrix or you will be unhappy (and so will your water molecules!)
        double3 relVelOH1 = velO - velH1;
        double3 relVelOH2 = velO - velH2;
        double3 relVelH1H2= velH1 - velH2;
        relativeVelocity.x = dot(relVelOH1,rOH1);
        relativeVelocity.y = dot(relVelOH2,rOH2);
        relativeVelocity.z = dot(relVelH1H2,rH1H2);


        if (DEBUG_BOOL) {
            printf("dc vals: %18.14f  %18.14f  %18.14f\n",
                   relativeVelocity.x, relativeVelocity.y,relativeVelocity.z);
        }

        double3 velCorrection = matrixVectorMultiply(fixRigidData.M1_inv,
                                                     fixRigidData.M2_inv,
                                                     fixRigidData.M3_inv,
                                                     relativeVelocity);
       

        // velocity corrections to apply to the atoms in the molecule 
        O_corr = (rOH1 * velCorrection.x + velCorrection.y * rOH2) * (-1.0 * fixRigidData.invMasses.z);
        H1_corr= (rOH1 * (-1.0) * velCorrection.x + rH1H2 * velCorrection.z) * (-1.0 * fixRigidData.invMasses.w);
        H2_corr= (rOH2 * (-1.0) * velCorrection.y - rH1H2 * velCorrection.z) * (-1.0 * fixRigidData.invMasses.w);
        
        velO  += O_corr;
        velH1 += H1_corr;
        velH2 += H2_corr;

        velCorrectionStored[idx*3]     = O_corr;
        velCorrectionStored[idx*3 + 1] = H1_corr;
        velCorrectionStored[idx*3 + 2] = H2_corr;

        velO_tmp = make_float3(velO);
        velH1_tmp= make_float3(velH1);
        velH2_tmp= make_float3(velH2);
        
        if (DEBUG_BOOL) {
            printf("settleProj routine after substracting corrections from derp!\n");
            printf("ow1 x,y,z derp: %18.14f   %18.14f   %18.14f\n",
                   velO.x, velO.y, velO.z);
            printf("hw2 x,y,z derp: %18.14f   %18.14f   %18.14f\n",
                   velH1.x, velH1.y, velH1.z);
            printf("hw3 x,y,z derp: %18.14f   %18.14f   %18.14f\n",
                   velH2.x, velH2.y, velH2.z);
            printf("\nEND settleProj routine!\n");
        }

        vs[idxO] = make_float4(velO_tmp.x, velO_tmp.y, velO_tmp.z, velO_whole.w);
        vs[idxH1]= make_float4(velH1_tmp.x,velH1_tmp.y,velH1_tmp.z,velH1_whole.w);
        vs[idxH2]= make_float4(velH2_tmp.x,velH2_tmp.y,velH2_tmp.z,velH2_whole.w);

    }

}

// implements the SETTLE algorithm for maintaining a rigid water molecule
template <class DATA, bool VIRIALS, bool DEBUG_BOOL>
__global__ void settlePositions(int4 *waterIds, float4 *xs, float4 *xs_0, 
                               float4 *vs, 
                               float4 *fs, Virial *virials, float4 *comOld, 
                               DATA fixRigidData, int nMolecules, 
                               float dt, float dtf,
                               int *idToIdxs, BoundsGPU bounds, int turn, double invdt) {
    int idx = GETIDX();
    if (idx < nMolecules) {

        // grab some data from our FixRigidData instance
        double inv2Rc = fixRigidData.canonicalTriangle.w; 
        double ra = fixRigidData.canonicalTriangle.x;
        double rb = fixRigidData.canonicalTriangle.y;
        double rc = fixRigidData.canonicalTriangle.z;
        double weightH = fixRigidData.weights.y;
        double mO = fixRigidData.weights.z;
        double mH = fixRigidData.weights.w;
        if (DEBUG_BOOL) {
            printf("wh: %18.14f\nrc: %18.14f\nra: %18.14f\nrb: %18.14f\nirc2: %18.14f\nmO: %18.14f\nmH: %18.14f\n",
                    weightH, rc, ra, rb, inv2Rc, mO, mH);
        }
        // so, our initial data from xs_0, accessed via idx;
        // -- this will form the basis of our X'Y'Z' system, the previous solution to the constraints
        double3 posO_initial = make_double3(xs_0[idx*3]);
        double3 posH1_initial= make_double3(xs_0[idx*3 + 1]);
        double3 posH2_initial= make_double3(xs_0[idx*3 + 2]);

        if (DEBUG_BOOL) {
            posO_initial = make_double3(17.4587872195011, 28.0685394926010, 19.7602624045003);
            posH1_initial = make_double3(18.3889844004616,  28.1063117245590, 19.5376903868836);
            posH2_initial = make_double3(17.2802653617548, 27.1368700632182, 19.8881446185112);
        
            printf("ow1 x,y,z b4: %18.14f   %18.14f   %18.14f\n",
                   posO_initial.x, posO_initial.y, posO_initial.z);
            printf("hw2 x,y,z b4: %18.14f   %18.14f   %18.14f\n",
                   posH1_initial.x, posH1_initial.y, posH1_initial.z);
            printf("hw3 x,y,z b4: %18.14f   %18.14f   %18.14f\n",
                   posH2_initial.x, posH2_initial.y, posH2_initial.z);
       
        }

        // get the molecule at this idx
        int4 atomsFromMolecule = waterIds[idx];

        // get the atom idxs
        // --- we used the variable 'idx' as an id above, in e.g. posO_initial
        //     and so we must treat it as the id here as well to access data 
        //     in a consistent manner
        int idxO = idToIdxs[atomsFromMolecule.x];
        int idxH1= idToIdxs[atomsFromMolecule.y];
        int idxH2= idToIdxs[atomsFromMolecule.z];
        
        // extract the whole positions
        float4 posO_whole = xs[idxO];
        float4 posH1_whole= xs[idxH1];
        float4 posH2_whole= xs[idxH2];

        // and the unconstrained, updated positions
        double3 posO = make_double3(posO_whole);
        double3 posH1= make_double3(posH1_whole);
        double3 posH2= make_double3(posH2_whole);
        
        
        
        
        // get the velocities from vs[] array
        float4 velO_whole = vs[idxO];
        float4 velH1_whole= vs[idxH1];
        float4 velH2_whole= vs[idxH2];
        
        // cast as double
        double3 velO = make_double3(velO_whole);
        double3 velH1= make_double3(velH1_whole);
        double3 velH2= make_double3(velH2_whole);
        
        if (DEBUG_BOOL) {

            posO = make_double3(17.4575271824439, 28.0674544952710, 19.7605025286681);
            posH1 = make_double3(18.3862875449773, 28.1295399140363, 19.5360517433464);
            posH2 = make_double3(17.3029620554269, 27.1308634077511, 19.8859719102520);

            velO = make_double3(-0.12600370572210, -0.10849973300457,  0.02401241678280)* 0.01;
            velH1= make_double3(-0.26968554843436,  2.32281894773213, -0.16386435371728)* 0.01;
            velH2= make_double3(2.26966936721430,  -0.60066554670720, -0.21727082592036)* 0.01;
       

            printf("ow1 x,y,z v : %18.14f   %18.14f   %18.14f\n",
                   velO.x, velO.y, velO.z);
            printf("hw2 x,y,z v : %18.14f   %18.14f   %18.14f\n",
                   velH1.x, velH1.y, velH1.z);
            printf("hw3 x,y,z v : %18.14f   %18.14f   %18.14f\n",
                   velH2.x, velH2.y, velH2.z);

            printf("ow1 x,y,z after: %18.14f  %18.14f  %18.14f\n",
                   posO.x, posO.y, posO.z);
            printf("hw2 x,y,z after: %18.14f  %18.14f  %18.14f\n",
                   posH1.x, posH1.y, posH1.z);
            printf("hw3 x,y,z after: %18.14f  %18.14f  %18.14f\n",
                   posH2.x, posH2.y, posH2.z);
        }
    
        // get the relative vectors OH1, OH2 for the initial triangle (solution from last step)
        double3 vectorOH1 = bounds.minImage(posH1_initial - posO_initial);
        double3 vectorOH2 = bounds.minImage(posH2_initial - posO_initial);
 
        // get the relative vectors for the unconstrained triangle
        double3 OH1_unconstrained = bounds.minImage(posH1 - posO);
        double3 OH2_unconstrained = bounds.minImage(posH2 - posO);

        // move the hydrogens to their minimum image distance w.r.t. the oxygen
        posH1 = posO + OH1_unconstrained;
        posH2 = posO + OH2_unconstrained;

        // the center of mass 'd1' in the paper
        double3 COM_d1 = (posO) + ( ( OH1_unconstrained + OH2_unconstrained ) * weightH);

        double3 posA1 = (OH1_unconstrained + OH2_unconstrained) * (-1.0 * weightH);
        double3 posB1 = posH1 - COM_d1;
        double3 posC1 = posH2 - COM_d1;

        // get our X'Y'Z' coordinate system
        double3 axis3 = cross(vectorOH1, vectorOH2);
        double3 axis1 = cross(posA1,axis3);
        double3 axis2 = cross(axis3,axis1);
       
        // normalize so that we have unit length basis vectors
        axis1 /= length(axis1);
        axis2 /= length(axis2);
        axis3 /= length(axis3);

        // rotate the relative vectors of the solved triangle
        double3 rotated_b0 = rotation(vectorOH1,axis1,axis2,axis3);
        double3 rotated_c0 = rotation(vectorOH2,axis1,axis2,axis3);
        
        // rotate our unconstrained update about the axes
        double3 rotated_a1 = rotation(posA1,axis1,axis2,axis3);
        double3 rotated_b1 = rotation(posB1,axis1,axis2,axis3);
        double3 rotated_c1 = rotation(posC1,axis1,axis2,axis3);

        double sinPhi = rotated_a1.z / ra; 
        double cosPhiSqr = 1.0 - (sinPhi * sinPhi);
        double cosPhi, cosPsiSqr, cosPsi, sinPsi;


        if (cosPhiSqr <= 0)
        {
            printf("cosPhiSqr <= 0 in settlePositions!\n");
        }
        else
        {
            cosPhi = sqrt(cosPhiSqr);
            sinPsi = (rotated_b1.z - rotated_c1.z) *  (inv2Rc / cosPhi);
            cosPsiSqr = 1.0 - (sinPsi * sinPsi);
            if (cosPsiSqr <= 0)
            {
                printf("cosPsiSqr <= 0 in settlePositions!\n");
            }
            else
            {
                cosPsi = sqrt(cosPsiSqr);
            }
        }

        double3 aPrime2 = make_double3(0.0, 
                                       ra * cosPhi,
                                       rb * sinPhi);

        double3 bPrime2 = make_double3(-1.0 * rc * cosPsi,
                                       -rb * cosPhi - rc * sinPsi * sinPhi,
                                       -rb * sinPhi + rc * sinPsi * cosPhi);

        double3 cPrime2 = make_double3(rc * cosPsi,
                                       -1.0 * rb * cosPhi + rc * sinPsi * sinPhi,
                                       -1.0 * rb * sinPhi - rc * sinPsi * cosPhi);

        
        double alpha = bPrime2.x * (rotated_b0.x - rotated_c0.x) +
                           bPrime2.y * (rotated_b0.y) + 
                           cPrime2.y * (rotated_c0.y);
    
        double beta  = bPrime2.x * (rotated_c0.y - rotated_b0.y) + 
                           bPrime2.y * (rotated_b0.x) + 
                           cPrime2.y * (rotated_c0.x);

        double gamma = (rotated_b0.x * rotated_b1.y) - 
                           (rotated_b1.x * rotated_b0.y) +
                           (rotated_c0.x * rotated_c1.y) - 
                           (rotated_c1.x * rotated_c0.y);

        double a2b2 = (alpha * alpha) + (beta * beta);
        double sinTheta = (alpha * gamma - (beta * sqrt(a2b2 - (gamma * gamma)))) / a2b2;

        double cosThetaSqr = 1.0 - (sinTheta * sinTheta);
        double cosTheta = sqrt(cosThetaSqr);

        double3 aPrime3 = make_double3(-aPrime2.y * sinTheta,
                                        aPrime2.y * cosTheta,
                                        rotated_a1.z);
    
        double3 bPrime3 = make_double3(bPrime2.x * cosTheta - bPrime2.y * sinTheta,
                                       bPrime2.x * sinTheta + bPrime2.y * cosTheta,
                                       bPrime2.z);

        double3 cPrime3 = make_double3(cPrime2.x * cosTheta - cPrime2.y * sinTheta,
                                       cPrime2.x * sinTheta + cPrime2.y * cosTheta,
                                       cPrime2.z);

        // get the transpose (inverse) of our rotation matrix from above
        double3 axis1T = make_double3(axis1.x, axis2.x, axis3.x);
        double3 axis2T = make_double3(axis1.y, axis2.y, axis3.y);
        double3 axis3T = make_double3(axis1.z, axis2.z, axis3.z);

        // rotate the solutions back to the original coordinate system
        double3 a_pos = rotation(aPrime3, axis1T, axis2T, axis3T);
        double3 b_pos = rotation(bPrime3, axis1T, axis2T, axis3T);
        double3 c_pos = rotation(cPrime3, axis1T, axis2T, axis3T);

        // add back COM to get final position of the atoms
        double3 oPosFinal = a_pos + COM_d1;
        double3 H1PosFinal= b_pos + COM_d1;
        double3 H2PosFinal= c_pos + COM_d1;

        // differential contributions to the velocities
        double3 dx_a = a_pos - posA1;
        double3 dx_b = b_pos - posB1;
        double3 dx_c = c_pos - posC1;

        // cast as float, then send to the global arrays
        float3 oPosNew = make_float3(oPosFinal);
        float3 H1PosNew= make_float3(H1PosFinal);
        float3 H2PosNew= make_float3(H2PosFinal);

        // set the positions in the global arrays as the new, solved positions
        xs[idxO] = make_float4(oPosNew.x, oPosNew.y, oPosNew.z,posO_whole.w);
        xs[idxH1]= make_float4(H1PosNew.x,H1PosNew.y,H1PosNew.z,posH1_whole.w);
        xs[idxH2]= make_float4(H2PosNew.x,H2PosNew.y,H2PosNew.z,posH2_whole.w);

        if (DEBUG_BOOL) {  
            printf("printing 'after' at end of settle routine\n");
            printf("ow1 x,y,z after: %18.14f  %18.14f  %18.14f\n",
                   oPosFinal.x, oPosFinal.y, oPosFinal.z);
            printf("hw2 x,y,z after: %18.14f  %18.14f  %18.14f\n",
                   H1PosFinal.x, H1PosFinal.y, H1PosFinal.z);
            printf("hw3 x,y,z after: %18.14f  %18.14f  %18.14f\n",
                   H2PosFinal.x, H2PosFinal.y, H2PosFinal.z);

        }
        // add the differential contributions to the velocity from settling the positions
        velO  += (dx_a * invdt);
        velH1 += (dx_b * invdt);
        velH2 += (dx_c * invdt);

        // cast as float
        float3 newVelO = make_float3(velO);
        float3 newVelH1= make_float3(velH1);
        float3 newVelH2= make_float3(velH2);

        if (DEBUG_BOOL) {
            printf("ow1 x,y,z v : %18.14f   %18.14f   %18.14f\n",
                   velO.x, velO.y, velO.z);
            printf("hw2 x,y,z v : %18.14f   %18.14f   %18.14f\n",
                   velH1.x, velH1.y, velH1.z);
            printf("hw3 x,y,z v : %18.14f   %18.14f   %18.14f\n",
                   velH2.x, velH2.y, velH2.z);
        }
        // set the velocities in global arrays as the new, solved velocities
        vs[idxO] = make_float4(newVelO,velO_whole.w);
        vs[idxH1]= make_float4(newVelH1,velH1_whole.w);
        vs[idxH2]= make_float4(newVelH2,velH2_whole.w);

    }
}

// so, removeNDF is called by instances of DataComputerTemperature on instantiation,
// which occurs (see IntegratorVerlet.cu, Integrator.cu) after 
// --- we must have already prepared FixRigid by the time a DataComputerTemperature requests NDF!
int FixRigid::removeNDF() {
    int ndf = 0;

    if (TIP4P) {
    // so, for each molecule, we have three constraints on the positions of the real atoms:
    // -- two OH bond lengths, and the angle between them (+3)
    // and we have three constraints on the virtual site (its position is completely defined) (+3)
    // 
    // in total, we have a reduction in configuration DOF of 6 per molecule if this is a 4-site model
        ndf = 6 * nMolecules;
        
    } else {
    
    // so, for each molecule, we have three constraints on the positions of the real atoms:
    // -- two OH bond lengths, and the angle between them (+3)
    // 
    // in total, we have a reduction in configuration DOF of 3 per molecule
        ndf = 3 * nMolecules;

    }

    return ndf;
}

// 
void FixRigid::populateRigidData() {
    // so at this point we have populated canonicalTriangle and sideLengths, which hold
    // the XYZ coordinate axes of the canonical triangle (see ref: Miyamoto) and the actual 
    // side lengths of the water molecule (OH, OH, HH), respectively

    // canonicalTriangle = double4(ra, rb, rc, 0.0)
    // sideLengths = double4(OH, OH, HH, OM)
    
    // some trigonometry here
    double cosC = fixRigidData.canonicalTriangle.z / fixRigidData.sideLengths.y;
    
    // isosceles triangles
    double cosB = cosC;
   
    //std::cout << "cosC found to be: " << cosC << " and cosB found to be: " << cosB << std::endl;
    // algebraic rearrangement of Law of Cosines, and solving for cosA:

    // a is the HH bond length, b is OH bond length, c is OH bond length (b == c)
    double a = fixRigidData.sideLengths.z;
    double b = fixRigidData.sideLengths.x;

    // b == c..
    double cosA = ((-1.0 * a * a ) + (b*b + b*b) ) / (2.0 * b * b);
    //std::cout << "cosA found to be: " << cosA << std::endl;
    /*
     * Set the members of fixRigidData to the computed values
     */
 
    /*
    fixRigidData.cosA = cosA;
    fixRigidData.cosC = cosC;
    fixRigidData.cosB = cosB;
    */

    // sum of the angles should be pi radians - if not, exit the simulation.
    double sumAngles = acos(cosA) + acos(cosB) + acos(cosC);
    if ( fabs(sumAngles - M_PI) > 0.000001) {
        printf("The sum of the angles for the triangle in SETTLE was found to be %3.10f radians, rather than pi radians. Aborting.\n",sumAngles);
        printf("cosA: %3.6f\ncosB: %3.6f\ncosC: %3.6f\nwith theta values of %f, %f %f\n", 
               cosA, cosB, cosC, acos(cosA), acos(cosB), acos(cosC));
        mdError("Sum of the interior angles of the triangle do not add up to pi radians in FixRigid.\n");
    }

    // also, get local variables of denominator, ma, mb, mc so that we can write the expressions concisely.

    // mass of Oxygen and Hydrogen respectively
    // --- these are initialized in ::set_fixed_sides()
    //double ma = fixRigidData.weights.z;
    //double mb = fixRigidData.weights.w;

    // first, we need to calculate the denominator expression, d
    /*
    double d = (1.0 / (2.0 * mb) ) * ( (2.0 * ( (ma+mb) * (ma+mb) ) ) + 
                                               (2.0 * ma * mb * cosA * cosB * cosC) - 
                                               (2.0 * mb * mb * cosA * cosA) - 
                                               (ma * (ma + mb) * ( (cosB * cosB) + (cosC * cosC) ) ) ) ;

    */
    //std::cout << "denominator expression 'd' found to be: " << d << std::endl;
    //printf("denominator expression evaluates to: %3.10f\n", d);

    // filling in the values for the tau expressions, so that we can do the velocity constraints
    // --- see Miyamoto, equations B2.
    //     Note that these are correction velocities w.r.t. the /solved/ positions at the end of the timestep,
    //     and so cosA, cosB, cosC etc. are all constants.  
    //     we are re-distributing the momenta s.t. the molecule is 
    //     only undergoing translational and rotational motion, and thus the COM has 6 DOF; the atoms do not.

    /*
    // these are just the factors, in order of listing, in equations B2 of Miyamoto
    double tauAB1 = (ma / d) * ( (2.0 * (ma + mb)) - (ma * cosC * cosC));
    double tauAB2 = (ma / d) * ( (mb * cosC * cosA) - ( (ma + mb) * cosB));
    double tauAB3 = (ma / d) * ( (ma * cosB * cosC) - (2.0 * mb * cosA));

    double tauBC1 = ( ((ma + mb) * (ma + mb)) - (mb * mb * cosA * cosA) ) / d;
    double tauBC2 = (ma / d) * ( (mb * cosA * cosB ) - ( (ma + mb) * cosC) );
    double tauBC3 = (ma / d) * ( (mb * cosC * cosA ) - ( (ma + mb) * cosB) );

    double tauCA1 = (ma / d) * ( (2.0 * (ma + mb) ) - ( ma * cosB * cosB) );
    double tauCA2 = (ma / d) * ( (ma * cosB * cosC) - (2.0 * mb * cosA));
    double tauCA3 = (ma / d) * ( (mb * cosA * cosB) - ( (ma + mb) * cosC));

    // and set the values to fixRigidData!! v. important...

    fixRigidData.tauAB1 = tauAB1;
    fixRigidData.tauAB2 = tauAB2;
    fixRigidData.tauAB3 = tauAB3;

    fixRigidData.tauBC1 = tauBC1;
    fixRigidData.tauBC2 = tauBC2;
    fixRigidData.tauBC3 = tauBC3;

    fixRigidData.tauCA1 = tauCA1;
    fixRigidData.tauCA2 = tauCA2;
    fixRigidData.tauCA3 = tauCA3;

    fixRigidData.denominator = d;
    */

    //std::cout << "invmH value: " << fixRigidData.invMasses.w << std::endl;

    //printf("invmH value: %18.14f\n",fixRigidData.invMasses.w);
    double invMH_normalized = fixRigidData.invMasses.w / fixRigidData.invMasses.z;

    double3 M1_tmp = make_double3(0.0, 0.0, 0.0);

    double HH = fixRigidData.sideLengths.z;
    //std::cout << "Using HH length of " << HH << std::endl;
    //printf("dHH:  %18.14f\n",HH);
    double OH = fixRigidData.sideLengths.x;
    //std::cout << "Using OH length of " << OH << std::endl;
    //printf("dOH:  %18.14f",OH);
    // [0,0]
    M1_tmp.x = 1.0 + ( invMH_normalized );
    // [0,1]
    M1_tmp.y = ( 1.0 - 0.5 * HH * HH  / (OH * OH) ) ;
    // [0,2]
    M1_tmp.z = invMH_normalized * 0.5 * HH / OH;

    double3 M2_tmp = make_double3(0.0, 0.0, 0.0);

    // [1,0]
    M2_tmp.x = M1_tmp.y;
    // [1,1]
    M2_tmp.y = M1_tmp.x;
    // [1,2]
    M2_tmp.z = M1_tmp.z;


    double3 M3_tmp = make_double3(0.0, 0.0, 0.0);
    // [2,0]
    M3_tmp.x = M1_tmp.z;
    // [2,1]
    M3_tmp.y = M2_tmp.z;
    // [2,2]
    M3_tmp.z = 2.0 * invMH_normalized;

    fixRigidData.M1 = M1_tmp;
    fixRigidData.M2 = M2_tmp;
    fixRigidData.M3 = M3_tmp;

    double3 M1_inv, M2_inv, M3_inv;

    // computes the inverse matrix of {M1;M2;M3} and stores in M1_inv, M2_inv, M3_inv;
    invertMatrix(M1_tmp,M2_tmp,M3_tmp,M1_inv,M2_inv,M3_inv);

    M1_inv *= fixRigidData.weights.z;
    M2_inv *= fixRigidData.weights.z;
    M3_inv *= fixRigidData.weights.z;

    fixRigidData.M1_inv = M1_inv;
    fixRigidData.M2_inv = M2_inv;
    fixRigidData.M3_inv = M3_inv;

    printf("M1_inv values: %18.14f    %18.14f    %18.14f\n",
            M1_inv.x, M1_inv.y, M1_inv.z);
    printf("M2_inv values: %18.14f    %18.14f    %18.14f\n",
            M2_inv.x, M2_inv.y, M2_inv.z);
    printf("M3_inv values: %18.14f    %18.14f    %18.14f\n",
            M3_inv.x, M3_inv.y, M3_inv.z);
    
    
    return;

}

bool FixRigid::postNVE_X() {

    float dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // Next,  do compute_SETTLE as usual on the (as-yet) unconstrained positions & velocities

    // from IntegratorVerlet
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;

    double invdt = (double) 1.0 /  (double(state->dt));

    int virialMode = state->dataManager.getVirialModeForTurn(state->turn);
    bool virials = ((virialMode == 1) or (virialMode == 2));

    // if we are computing virials this turn, do so now
    // TODO: this is being done /before/ forces are computed - so, the virials get wiped!
    // XXX XXX: we'll need a local gpuarray of virials for temporary storage. TBD.
    if (virials) {
        settlePositions<FixRigidData,true,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt);

    } else {
        settlePositions<FixRigidData,false,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt);

    }

    if (TIP4P) {
    
        setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds, fixRigidData);
    
    }

    return true;
}




void FixRigid::set_fixed_sides() {
   
    // we already have the bond lengths from 'setStyleBondLengths'...

    // we have not yet set the 'weights' attribute for our fixRigidData instance.
    // do so now.

    mdAssert(waterIds.size() > 0,"There needs to be at least one water in FixRigid for thsi fix to be activated!");
    int4 firstMolecule = waterIds[0];
    int id_O1 = firstMolecule.x;
    int id_H1 = firstMolecule.y;
    //double massO = 15.999400000;
    double massO = state->atoms[id_O1].mass;
    std::cout << "massO found to be " << massO << std::endl;
    //double massH = 1.00800000;
    double massH = state->atoms[id_H1].mass;
    std::cout << "massH1 found to be " << massH << std::endl;
    double massWater = massO + (2.0 * massH);

    std::cout << "massWater found to be " << massWater << std::endl;


    fixRigidData.weights.x = massO / massWater;
    fixRigidData.weights.y = massH / massWater;
    fixRigidData.weights.z = massO;
    fixRigidData.weights.w = massH;

    double4 invWeights = make_double4( 1.0 / (massO / massWater),
                                       1.0 / (massH / massWater),
                                       1.0 / massO,
                                       1.0 / massH);

    fixRigidData.invMasses = invWeights;

    //float4 fixRigidData.sideLengths is as OH, OH, HH [, OM]...
    double rc = 0.5 * fixRigidData.sideLengths.z;

    // OH bond length
    double OH = fixRigidData.sideLengths.x;
    // so, a little bit of trig
    double raPlusRb = sqrt( (OH * OH) - (rc * rc));

    // ra is just the distance along the y-axis from Oxygen to COM..
    // -- compute COM with oxygen at origin, then compute COM and get distance in y-dir (molecule laying in XY plane)
    // -- we know the apex angle HOH due to having the OH, OH, HH bond lengths. 
    //    basic trig to get theta, then compute COM place the two hydrogens at +/- (0.5 theta) along the y axis (3pi/2)
    //    compute COM using weights & double3 position arithmetic average

    double3 pos_O = make_double3(0,0,0);
    
    double halftheta = atan(rc/raPlusRb);

    double h1theta = (3.0 * M_PI / 2.0) + halftheta;
    double h2theta = (3.0 * M_PI / 2.0) - halftheta;
    // and multiply these by the OH bond lengths for this model
    double3 pos_H1 = make_double3( cos(h1theta), sin(h1theta), 0)  * OH;
    double3 pos_H2 = make_double3( cos(h2theta),  sin(h2theta), 0) * OH;

    double3 com = (pos_O * fixRigidData.weights.x) + ( (pos_H1 + pos_H2) * fixRigidData.weights.y);
    
    // take diff.y as ra
    double3 diff = (pos_O - com);

    double ra = diff.y;
    // finally, subtract ra from raPlusRb to get rb.
    double rb = raPlusRb - ra;

    std::cout << "ra: " << ra << "; rb: " << rb << "; rc: " << rc << std::endl;


    double raVal = 2.0 * massH * sqrt(OH * OH - (0.5 * fixRigidData.sideLengths.z * 0.5 * fixRigidData.sideLengths.z) ) / (massWater);
    double rbVal = sqrt(OH * OH - (0.5 * fixRigidData.sideLengths.z * 0.5 * fixRigidData.sideLengths.z)) - raVal;
    double inv2Rc = 1.0 / fixRigidData.sideLengths.z;

    std::cout << "raVal: " << raVal << "; rbVal: " << rbVal << "; inv2Rc: " << inv2Rc << std::endl;
    fixRigidData.canonicalTriangle = make_double4(ra,rb,rc,inv2Rc);

}




void FixRigid::setStyleBondLengths() {
    
    if (TIP4P) {
        // 4-site models here

        if (style == "TIP4P/2005") {
            
            sigma_O = 3.15890000;
            r_OH = 0.95720000000;
            r_HH = 1.51390000000;
            r_OM = 0.15460000000;
            // see q-TIP4P/f paper by Manolopoulos et. al., eq. 2; this quantity can be computed from an arbitrary
            // TIP4P/2005 molecule fairly easily and is consistent with the above r_OH, r_HH, r_OM values
            gamma = 0.73612446364836; 
        } else {
            
            mdError("Only TIP3P and TIP4P/2005 are supported in setStyleBondLengths at the moment.\n");

        }

    } else if (TIP3P) {
        // 3-site models here
        
        // set r_OM to 0.0 for completeness
        r_OM = 0.00000000000;
        gamma = 0.0000;
        if (style == "TIP3P") {

            sigma_O = 3.15890000;
            r_OH = 0.95720000000;
            r_HH = 1.51390000000;
        } else {

            mdError("Only TIP3P and TIP4P/2005 are supported in setStyleBondLengths at the moment.\n");

        }
    } else {

        mdError("Neither 3-site nor 4-site model currently selected in FixRigid.\n");

    }

    if (state->units.unitType == UNITS::LJ) {

        // how do we enforce that consistent scaling is achieved here? 
        // where consistent is understood to be w.r.t. how the user scales the LJ interactions, charge-charge interactions, etc.
        // since this is modular - i.e., FixRigid just maintains the geometry and partitions M-site forces (if necessary);
        // ---- needs to be given more thought
        //r_OH /= sigma_O;
        //r_OM /= sigma_O;
        //r_HH /= sigma_O;
        mdError("Currently, real units must be used for simulations of rigid water models.\n");
    }

    fixedSides = make_double4(r_OH, r_OH, r_HH, r_OM);

    // set the sideLengths variable in fixRigidData to the fixedSides values
    fixRigidData.sideLengths = fixedSides;
    fixRigidData.gamma = gamma;
    double invOM = 0.0;
    if (TIP4P) invOM = 1.0 / r_OM;

    double4 invSideLengths = make_double4(1.0/r_OH, 1.0 / r_OH, 1.0 / r_HH, invOM);
    fixRigidData.invSideLengths = invSideLengths;
    return;

}

std::string FixRigid::restartChunk(std::string format) {
    std::stringstream ss;
    
    ss << "<atomsInMolecule n=\"" << waterIds.size() << "\">\n";
    for (int4 &atomIds : waterIds) {
        ss << atomIds.x << " " << atomIds.y << " " << atomIds.z << " " << atomIds.w << "\n";
    }
    ss << "</atomsInMolecule>\n";

    ss << "<style type=\'" << style << "\'>\n";
    ss << "</style>\n";
    if (TIP4P) {
        ss << "<model fourSite=\'true\'/>\n";
    } else {
        ss << "<model fourSite=\'false\'/>\n";
    }

    return ss.str();
}

bool FixRigid::readFromRestart() {
    auto restData = getRestartNode();

    if (restData) {

        auto curr_param = restData.first_child();
        while (curr_param) {
            std::string tag = curr_param.name();
            if (tag == "atomsInMolecule") {
                int n = boost::lexical_cast<int>(curr_param.attribute("n").value());
                std::vector<int4> atomsInMolecule(n);
                // we need to pass this data to atomsInMolecule, which will then be assigned to waterIds;
                // and while we are here, we may as well make the bonds
                // -- note: even in a 3-site model (e.g. TIP3P), we use int4's. so, there shouldn't be an issue here, although some xml file size bloat occurs. Something to address later.
                xml_assignValues<int, 4>(curr_param, [&] (int i, int *vals) {
                                            //int id = vals[3];
                                            atomsInMolecule[i] = make_int4(vals[0], vals[1], vals[2], vals[3]);
                                            });

                // and assign our class member 'waterIds' the data denoted by atomsInMolecule
                waterIds = atomsInMolecule;
            } else if (tag == "style") {
                std::string thisStyle = boost::lexical_cast<std::string>(curr_param.attribute("type").value());
                style = thisStyle;
                printf("In FixRigid::readFromRestart(), found style %s\n", style.c_str());
            } else if (tag == "model") {
                bool fourSite = boost::lexical_cast<bool>(curr_param.attribute("fourSite").value());
                // set our class member boolean flags TIP3P & TIP4P
                TIP4P = fourSite;
                TIP3P = !fourSite;
            }
            curr_param = curr_param.next_sibling();
        }

    }

    
    for (int i = 0; i < waterIds.size(); i++) {
        Bond bondOH1;
        Bond bondOH2;
        Bond bondHH;
        bondOH1.ids = { {waterIds[i].x,waterIds[i].y} };
        bondOH2.ids = { {waterIds[i].x,waterIds[i].z} };
        bondHH.ids =  { {waterIds[i].y,waterIds[i].z} };

        bonds.push_back(bondOH1);
        bonds.push_back(bondOH2);
        bonds.push_back(bondHH);
        if (TIP4P) {
            Bond bondOM;
            bondOM.ids = { {waterIds[i].x, waterIds[i].w } };
            bonds.push_back(bondOM);
        }
    }

    std::cout << "There are " << waterIds.size() << " molecules read in from the restart file and " << bonds.size() << " bonds were made in FixRigid.\n";
    return true;
}

void FixRigid::setStyle(std::string style_) {
    if (style_ == "TIP4P/LONG") {
        style = style_;
    } else if (style_ == "TIP4P/2005") {
        style = style_;
    } else if (style_ == "TIP4P") {
        style = style_;
    } else if (style_ == "TIP3P") {
        style = style_;
    } else if (style == "TIP3P/LONG") {
        style = style_;
    } else {
        mdError("Supported style arguments are \"TIP4P\", \"TIP4P/2005\", \"TIP4P/LONG\", \"TIP3P\", \"TIP3P/LONG\"");
    };

}

void FixRigid::assignNDF() {
   
    // if this is TIP3P, assign NDF of 2 to each atom in a given molecule (distribute the reduction equally)
    if (TIP3P) {
        for (int i = 0; i < waterIds.size(); i++) {
            int4 thisMolIds = waterIds[i];
            int idO = thisMolIds.x;
            int idH1= thisMolIds.y;
            int idH2= thisMolIds.z;
            Atom &O = state->atoms[state->idToIdx[idO]];
            Atom &H1= state->atoms[state->idToIdx[idH1]];
            Atom &H2= state->atoms[state->idToIdx[idH2]];
            O.setNDF(2);
            H1.setNDF(2);
            H2.setNDF(2);
        }
    }
    // if this is 4 site model, assign NDF of 2 to each real atom, and 0 to the M-site
    if (TIP4P) {
        
        for (int i = 0; i < waterIds.size(); i++) {
            int4 thisMolIds = waterIds[i];
            int idO = thisMolIds.x;
            int idH1= thisMolIds.y;
            int idH2= thisMolIds.z;
            int idM = thisMolIds.w;
            Atom &O = state->atoms[state->idToIdx[idO]];
            Atom &H1= state->atoms[state->idToIdx[idH1]];
            Atom &H2= state->atoms[state->idToIdx[idH2]];
            Atom &Msite = state->atoms[state->idToIdx[idM]];
            O.setNDF(2);
            H1.setNDF(2);
            H2.setNDF(2);
            Msite.setNDF(0);
        }
    }
}

//returns ids of atoms that belong to a constrained group
std::vector<int> FixRigid::getRigidAtoms() {

    std::vector<int> atomsToReturn;
    // by now, this is prepared. so, we know if it is a 3-site or 4-site model.
    if (TIP3P) {
        atomsToReturn.reserve(3 * nMolecules);
    } else {
        atomsToReturn.reserve(4 * nMolecules);
    }

    if (TIP3P) {
        for (int i = 0; i < waterIds.size(); i++ ) {
            int4 theseAtomIds = waterIds[i];
            int idO = theseAtomIds.x;
            int idH1= theseAtomIds.y;
            int idH2= theseAtomIds.z;
            atomsToReturn.push_back(idO);
            atomsToReturn.push_back(idH1);
            atomsToReturn.push_back(idH2);
        }
    } else {
        for (int i = 0; i < waterIds.size(); i++ ) {
            int4 theseAtomIds = waterIds[i];
            int idO = theseAtomIds.x;
            int idH1= theseAtomIds.y;
            int idH2= theseAtomIds.z;
            int idM = theseAtomIds.w;
            atomsToReturn.push_back(idO);
            atomsToReturn.push_back(idH1);
            atomsToReturn.push_back(idH2);
            atomsToReturn.push_back(idM);
        }
    }

    return atomsToReturn;
}

void FixRigid::scaleRigidBodies(float3 scaleBy, uint32_t groupTag) {

    // so, pass the objects by molecule, displace the COM, and then apply the difference to the individual atoms

    // consider: we might have different groups of water molecules (e.g., simulation of solid-liquid interface).
    // so, use the groupTags.
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    if (groupTag == 1) {
        if (TIP4P) {
            rigid_scaleSystem_cu<FixRigidData, true><<<NBLOCK(nMolecules),PERBLOCK>>>(waterIdsGPU.data(),
                                                                  gpd.xs(activeIdx),
                                                                  gpd.idToIdxs.d_data.data(),
                                                                  bounds.lo,
                                                                  bounds.rectComponents,
                                                                  bounds,
                                                                  scaleBy,
                                                                  fixRigidData,
                                                                  nMolecules);
        } else { 

            rigid_scaleSystem_cu<FixRigidData, false><<<NBLOCK(nMolecules),PERBLOCK>>>(waterIdsGPU.data(),
                                                                  gpd.xs(activeIdx),
                                                                  gpd.idToIdxs.d_data.data(),
                                                                  bounds.lo,
                                                                  bounds.rectComponents,
                                                                  bounds, 
                                                                  scaleBy,
                                                                  fixRigidData,
                                                                  nMolecules);

        }

    } else {
        if (TIP4P) {
            rigid_scaleSystemGroup_cu<FixRigidData,true><<<NBLOCK(nMolecules),PERBLOCK>>>(waterIdsGPU.data(),
                                                                  gpd.xs(activeIdx),
                                                                  gpd.idToIdxs.d_data.data(),
                                                                  bounds.lo,
                                                                  bounds.rectComponents,
                                                                  bounds, 
                                                                  scaleBy,
                                                                  fixRigidData,
                                                                  nMolecules, 
                                                                  groupTag,
                                                                  gpd.fs(activeIdx)
                                                                  );
        } else {
            rigid_scaleSystemGroup_cu<FixRigidData,false><<<NBLOCK(nMolecules),PERBLOCK>>>(waterIdsGPU.data(),
                                                                  gpd.xs(activeIdx),
                                                                  gpd.idToIdxs.d_data.data(),
                                                                  bounds.lo,
                                                                  bounds.rectComponents,
                                                                  bounds, 
                                                                  scaleBy,
                                                                  fixRigidData,
                                                                  nMolecules, 
                                                                  groupTag,
                                                                  gpd.fs(activeIdx)
                                                                  );
        }
    }
}

void FixRigid::createRigid(int id_a, int id_b, int id_c, int id_d) {
    TIP4P = true;
    int4 waterMol = make_int4(0,0,0,0);
    Vector a = state->idToAtom(id_a).pos;
    Vector b = state->idToAtom(id_b).pos;
    Vector c = state->idToAtom(id_c).pos;
    Vector d = state->idToAtom(id_d).pos;

    double ma = state->idToAtom(id_a).mass;
    double mb = state->idToAtom(id_b).mass;
    double mc = state->idToAtom(id_c).mass;
    double md = state->idToAtom(id_d).mass;


    double ims = 1.0 / (ma + mb + mc + md);
    float4 ims4 = make_float4(0.0f, 0.0f, 0.0f, float(ims));
    invMassSums.push_back(ims4);

    bool ordered = true;
    if (! (ma > mb && ma > mc)) ordered = false;
    
    if (! (mb == mc) ) ordered = false;
    if (! (mb > md) )  ordered = false;
    
    if (! (ordered)) {
        printf("Found masses O, H, H, M in order: %f %f %f %f\n", ma, mb, mc, md);
    }
    if (! (ordered)) mdError("Ids in FixRigid::createRigid must be as O, H1, H2, M");

    /*
    printf("adding ids %d %d %d %d\n", id_a, id_b, id_c, id_d);
    printf("with masses %f %f %f %f\n", ma, mb, mc, md);
    printf("position id_a: %f %f %f\n", a[0], a[1], a[2]);
    printf("position id_b: %f %f %f\n", b[0], b[1], b[2]);
    printf("position id_c: %f %f %f\n", c[0], c[1], c[2]);
    printf("position id_c: %f %f %f\n ", d[0], d[1], d[2]);
    */
    waterMol = make_int4(id_a, id_b, id_c, id_d);
    
    waterIds.push_back(waterMol);

    Bond bondOH1;
    Bond bondOH2;
    Bond bondHH;
    Bond bondOM;

    bondOH1.ids = { {waterMol.x,waterMol.y} };
    bondOH2.ids = { {waterMol.x,waterMol.z} };
    bondHH.ids =  { {waterMol.y,waterMol.z} };
    bondOM.ids =  { {waterMol.x,waterMol.w} };

    bonds.push_back(bondOH1);
    bonds.push_back(bondOH2);
    bonds.push_back(bondHH);
    bonds.push_back(bondOM);

    // finally, if the state->rigidBodies flag is not yet tripped, set it to true
    state->rigidBodies = true;

}

void FixRigid::createRigid(int id_a, int id_b, int id_c) {
    TIP3P = true;
    int4 waterMol = make_int4(0,0,0,0);
    Vector a = state->idToAtom(id_a).pos;
    Vector b = state->idToAtom(id_b).pos;
    Vector c = state->idToAtom(id_c).pos;

    double ma = state->idToAtom(id_a).mass;
    double mb = state->idToAtom(id_b).mass;
    double mc = state->idToAtom(id_c).mass;
    double ims = 1.0 / (ma + mb + mc);
    float4 ims4 = make_float4(0.0f, 0.0f, 0.0f, float(ims));
    invMassSums.push_back(ims4);

    // this discovers the order of the id's that was passed in, i.e. OHH, HOH, HHO, etc.
    float det = a[0]*b[1]*c[2] - a[0]*c[1]*b[2] - b[0]*a[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2] - c[0]*b[1]*a[2];
    if (state->idToAtom(id_a).mass == state->idToAtom(id_b).mass) {
    waterMol = make_int4(id_c,id_a,id_b,0);
    if (det < 0) {
      waterMol = make_int4(id_c,id_b,id_a,0);
    }
    }
    else if (state->idToAtom(id_b).mass == state->idToAtom(id_c).mass) {
    waterMol = make_int4(id_a,id_b,id_c,0);
    if (det < 0) {
      waterMol = make_int4(id_a,id_c,id_b,0);
    }
    }
    else if (state->idToAtom(id_c).mass == state->idToAtom(id_a).mass) {
    waterMol = make_int4(id_b,id_c,id_a,0);
    if (det < 0) {
      waterMol = make_int4(id_b,id_a,id_c,0);
    }
    } else {
    assert("waterMol set" == "true");
    }
    waterIds.push_back(waterMol);
    Bond bondOH1;
    Bond bondOH2;
    Bond bondHH;
    bondOH1.ids = { {waterMol.x,waterMol.y} };
    bondOH2.ids = { {waterMol.x,waterMol.z} };
    bondHH.ids = { {waterMol.y,waterMol.z} };
    bonds.push_back(bondOH1);
    bonds.push_back(bondOH2);
    bonds.push_back(bondHH);
    
    state->rigidBodies = true;
}

bool FixRigid::postNVE_V() {


    return true;
}

bool FixRigid::prepareForRun() {
    
    nMolecules = waterIds.size();
    // consider: in postRun(), we set rigidBodies to false;
    // if another run command is issued without adding another water molecule, it will remain false, 
    // and that will cause problems.  So, set it to true here as well.  Is this too late though? 
    // ---- I think it is not too late to trip this flag here.
    if (nMolecules > 0) {
        state->rigidBodies = true;
    }
    // cannot have more than one water model present
    if (TIP3P && TIP4P) {
        mdError("An attempt was made to use both 3-site and 4-site models in a simulation");
    }

    if (!(TIP3P || TIP4P)) {
        mdError("An attempt was made to use neither 3-site nor 4-site water models with FixRigid in simulation.");
    }

    // set the specific style of the fix
    if (style == "DEFAULT") {
        if (TIP3P) {
            style = "TIP3P";
        }
        if (TIP4P) {
            style = "TIP4P/2005";
        }
    }

    // an instance of FixRigidData
    fixRigidData = FixRigidData();
    setStyleBondLengths();
    set_fixed_sides();
    populateRigidData();

    printf("number of molecules in waterIds: %d\n", nMolecules);
    waterIdsGPU = GPUArrayDeviceGlobal<int4>(nMolecules);
    waterIdsGPU.set(waterIds.data());


    velCorrectionStored = GPUArrayDeviceGlobal<double3>(3*nMolecules);
    
    xs_0 = GPUArrayDeviceGlobal<float4>(3*nMolecules);
    com = GPUArrayDeviceGlobal<float4>(nMolecules);
    constraints = GPUArrayDeviceGlobal<bool>(nMolecules);
    com.set(invMassSums.data());

    float dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    
    int virialMode = state->dataManager.getVirialModeForTurn(state->turn);
    bool virials = ((virialMode == 1) or (virialMode == 2));
    
    compute_COM<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                                gpd.xs(activeIdx), 
                                                                gpd.vs(activeIdx), 
                                                                gpd.idToIdxs.d_data.data(), 
                                                                nMolecules, com.data(), 
                                                                bounds, fixRigidData);

    
    save_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                nMolecules, gpd.idToIdxs.d_data.data());

    double invdt = (double) 1.0 /  (double(state->dt));

    // iterate over all atoms and reduce their configurational DOF accordingly as the atom struct
    assignNDF();
    // so; solve the positions (if we have good initial conditions, nothing happens here)

    if (solveInitialConstraints) {
        if (virials) {
            settlePositions<FixRigidData,true, false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                        xs_0.data(), gpd.vs(activeIdx),  
                                                        gpd.fs(activeIdx), 
                                                        gpd.virials.d_data.data(),
                                                        com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                        gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt);
        } else {

            settlePositions<FixRigidData,false, false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                        xs_0.data(), gpd.vs(activeIdx), 
                                                        gpd.fs(activeIdx), 
                                                        gpd.virials.d_data.data(),
                                                        com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                        gpd.idToIdxs.d_data.data(), bounds, (int) state->turn, invdt);
        }

        // I hope the M-site was in the proper positions in the initial configuration... otherwise we have bad forces

        // so;
        // in the event that this is a /restart/ - we actually do not want to compute forces on M-sites and distribute 
        // them; because, this would be distributing them /again/ (they were distributed before the prior simulation 
        // concluded).  
        // In the event that this is not a restart, our initial velocites were sampled from some distribution, 
        // in which case distribution of the M-site forces would disrupt the distribution that was specified.
        // so, don't distribute the M-site forces;
        // BUT: if we are doing NPT simulation, we still need an accurate measure of the Virials!
        //      So, regardless, these must be distributed.
        if (TIP4P) {
            // if we need the virials initially, distribute them; do not distribute forces.
            if (virials) {
                distributeMSite<true,false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), 
                                                                              gpd.xs(activeIdx), 
                                                             gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                             gpd.virials.d_data.data(),
                                                             nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
                
            }  
            // just to be redundant - set the M-site at its assigned position
            setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds,fixRigidData);
        }


        settleVelocities<FixRigidData,false, false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), velCorrectionStored.data(), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
        // and remove the COMV we may have acquired from settling the velocity constraints just now.
        GPUArrayDeviceGlobal<float4> sumMomentum = GPUArrayDeviceGlobal<float4>(2);

        // should probably make removal of COMV optional
        CUT_CHECK_ERROR("Error occurred during solution of velocity constraints.");
        sumMomentum.memset(0);
        int warpSize = state->devManager.prop.warpSize;

        float3 dimsFloat3 = make_float3(1.0, 1.0, 1.0);
        accumulate_gpu<float4, float4, SumVectorXYZOverW, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms / (double) N_DATA_PER_THREAD), PERBLOCK, N_DATA_PER_THREAD*PERBLOCK*sizeof(float4)>>>
                (
                 sumMomentum.data(),
                 gpd.vs(activeIdx),
                 nAtoms,
                 warpSize,
                 SumVectorXYZOverW()
                );
        rigid_remove_COMV<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, gpd.vs(activeIdx), sumMomentum.data(), dimsFloat3);
        CUT_CHECK_ERROR("Removal of COMV in FixRigid failed.");
        
        // adjust the initial velocities to conform to our velocity constraints
        // -- here, we preserve the COMV of the system, while imposing strictly translational motion on the molecules
        //    - we use shared memory to compute center of mass velocity of the group, allowing for one kernel call
        
        // validate that we have good initial conditions
        SAFECALL((validateConstraints<FixRigidData> <<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), 
                                                               gpd.xs(activeIdx), gpd.vs(activeIdx), 
                                                               nMolecules, bounds, fixRigidData, 
                                                               constraints.data(), state->turn)));
        CUT_CHECK_ERROR("Validation of constraints failed in FixRigid.");
    }

    prepared = true;
    return prepared;
}

bool FixRigid::stepInit() {
    
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    
    // save the positions, velocities, forces from the previous, fully updated turn in to our local arrays
    // ---- we really only need the positions here
    save_prev_val<<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), xs_0.data(), 
                                                nMolecules, gpd.idToIdxs.d_data.data());


    return true;
}

bool FixRigid::stepFinal() {
    float dt = state->dt;
    GPUData &gpd = state->gpd;
    int activeIdx = gpd.activeIdx();
    BoundsGPU &bounds = state->boundsGPU;
    int nAtoms = state->atoms.size();
    // first, unconstrained velocity update continues: distribute the force from the M-site
    //        and integrate the velocities accordingly.  Update the forces as well.
    // finally, solve the velocity constraints on the velocities.
    // from IntegratorVerlet
    float dtf = 0.5f * state->dt * state->units.ftm_to_v;
    
    int virialMode = state->dataManager.getVirialModeForTurn(state->turn);
    bool virials = ((virialMode == 1) or (virialMode == 2));


    if (TIP4P) {
        if (virials) {
            distributeMSite<true,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                         gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                         gpd.virials.d_data.data(),
                                                         nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
            
        } else { 
        
            distributeMSite<false,true><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                         gpd.vs(activeIdx),  gpd.fs(activeIdx),
                                                         gpd.virials.d_data.data(),
                                                         nMolecules, gamma, dtf, gpd.idToIdxs.d_data.data(), bounds);
        }
    }

    if (virials) {
        settleVelocities<FixRigidData,true, false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), velCorrectionStored.data(), 
                                                    gpd.fs(activeIdx), 
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);
    } else {

        settleVelocities<FixRigidData,false, false><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.xs(activeIdx), 
                                                    xs_0.data(), gpd.vs(activeIdx), velCorrectionStored.data(), 
                                                    gpd.fs(activeIdx),
                                                    gpd.virials.d_data.data(),
                                                    com.data(), fixRigidData, nMolecules, dt, dtf, 
                                                    gpd.idToIdxs.d_data.data(), bounds, (int) state->turn);

    }
    // set the MSite; note that this is not required for proper integration/dynamics; rather,
    // we do this extraneous call so that, in the even the configuration is printed out, the Msite is in the 
    // correct position.
    
    if (TIP4P) {
        setMSite<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>>(waterIdsGPU.data(), gpd.idToIdxs.d_data.data(), gpd.xs(activeIdx), nMolecules, bounds,fixRigidData);
    }


    validateConstraints<FixRigidData><<<NBLOCK(nMolecules), PERBLOCK>>> (waterIdsGPU.data(), 
                                                                                   gpd.idToIdxs.d_data.data(), 
                                                                                   gpd.xs(activeIdx), 
                                                                                   gpd.vs(activeIdx), 
                                                                                   nMolecules, bounds, 
                                                                                   fixRigidData, 
                                                                                   constraints.data(), 
                                                                                   (int) state->turn);
    return true;
}


// postRun is primarily for re-setting local and global flags;
// in this case, tell state that there are no longer rigid bodies
bool FixRigid::postRun() {
    prepared = false;
    state->rigidBodies = false;
    return true;
}



// export the overloaded function
void (FixRigid::*createRigid_x1) (int, int, int)      = &FixRigid::createRigid;
void (FixRigid::*createRigid_x2) (int, int, int, int) = &FixRigid::createRigid;

void export_FixRigid() 
{
  py::class_<FixRigid, boost::shared_ptr<FixRigid>, py::bases<Fix> > 
      ( 
		"FixRigid",
		py::init<boost::shared_ptr<State>, std::string, std::string>
	    (py::args("state", "handle", "groupHandle")
         )
        )
    .def("createRigid", createRigid_x1,
	    (py::arg("id_a"), 
         py::arg("id_b"), 
         py::arg("id_c")
         )
	 )
    .def("createRigid", createRigid_x2,
        (py::arg("id_a"), 
         py::arg("id_b"),  
         py::arg("id_c"), 
         py::arg("id_d"))
     )
    .def("setStyle", &FixRigid::setStyle,
         py::arg("style")
        )
	.def_readwrite("printing", &FixRigid::printing)
    .def_readwrite("solveInitialConstraints", &FixRigid::solveInitialConstraints)
    ;
}



