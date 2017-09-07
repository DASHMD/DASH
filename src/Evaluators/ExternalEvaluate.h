#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"


template <class EVALUATOR, bool COMPUTE_VIRIALS>
__global__ void compute_force_external(int nAtoms,float4 *xs, float4 *fs, uint groupTag,Virial *__restrict__ virials, EVALUATOR eval) 
        {
	int idx = GETIDX();
	if (idx < nAtoms) {
	    float4 forceWhole = fs[idx];
	    uint groupTagAtom = * (uint *) &forceWhole.w;
	    // Check if atom is part of group affected by external potential
	    if (groupTagAtom & groupTag) {
            //Virial virialSum(0, 0, 0, 0, 0, 0);
	        float4 posWhole = xs[idx];
	        float3 pos      = make_float3(posWhole);
            float3 force    = eval.force( pos );      // compute the force due to ext. potential!
            float4 f        = fs[idx];
            f               = f + force;
            fs[idx]         = f;
            //if (COMPUTE_VIRIALS) {
            //    computeVirial(virialSum,force,pos);
            //    virials[idx] += virialSum;
            //}
            }
	    }
	}


template <class EVALUATOR>
__global__ void compute_energy_external(int nAtoms,float4 *xs, float4 *fs, float *perParticleEng, uint groupTag, EVALUATOR eval) 
        {
	int idx = GETIDX();
	if (idx < nAtoms) {
	  float4 forceWhole = fs[idx];
	  uint groupTagAtom = * (uint *) &forceWhole.w;
	  // Check if atom is part of group affected by external potential
	  if (groupTagAtom & groupTag) {
	    float4 posWhole = xs[idx];
	    float3 pos      = make_float3(posWhole);
            float  uext     = eval.energy( pos );      // compute the energy due to ext. potential!
            perParticleEng[idx] += uext;
            }
	  }
	}

