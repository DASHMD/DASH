#include "BoundsGPU.h"
#include "cutils_func.h"
#include "helpers.h"


template <class EVALUATOR, bool COMPUTE_VIRIALS>
__global__ void compute_wall_iso(int nAtoms,float4 *xs, float4 *fs,float3 origin,
		float3 forceDir,  uint groupTag, EVALUATOR eval) {


	int idx = GETIDX();
	if (idx < nAtoms) {
		float4 forceWhole = fs[idx];
		uint groupTagAtom = * (uint *) &forceWhole.w;
		// if this atom is assigned to the group affected by this wall fix, then..
		if (groupTagAtom & groupTag) {
			float4 posWhole = xs[idx];
			float3 pos = make_float3(posWhole);
			float3 particleDist = pos - origin;
			float projection = dot(particleDist, forceDir);
			float magProj = cu_abs(projection);
            float3 force = eval.force(magProj, forceDir);

            float4 f = fs[idx];
            if (projection >= 0) {
                f = f + force;
            } else {
                printf("Atom pos %f %f %f wall origin %f %f %f\n", pos.x, pos.y, pos.z, origin.x, origin.y, origin.z);
                assert(projection>0); // projection should be greater than 0, otherwise
                // the wall is ill-defined (forceDir pointing out of box)
                //
            }
            fs[idx] = f;


			 
		}
	}
}





