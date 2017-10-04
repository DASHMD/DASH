#define SMALL 0.0001f
template <class ANGLETYPE, class EVALUATOR, bool COMPUTEVIRIALS>
__global__ void compute_force_angle(int nAtoms, float4 *xs, float4 *forces, int *idToIdxs, AngleGPU *angles, int *startstops, BoundsGPU bounds, ANGLETYPE *parameters_arg, int nParameters, Virial *__restrict__ virials, bool usingSharedMemForParams, EVALUATOR evaluator) {

    int idx = GETIDX();
    extern __shared__ char all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    AngleGPU *angles_shr = (AngleGPU *) all_shr;
    int sizeAngles = (idxEndCopy - idxBeginCopy) * sizeof(AngleGPU);
    copyToShared<AngleGPU>(angles + idxBeginCopy, angles_shr, idxEndCopy - idxBeginCopy);
    ANGLETYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (ANGLETYPE *) (all_shr + sizeAngles);
        copyToShared<ANGLETYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();
    if (idx < nAtoms) {
        //printf("going to compute %d\n", idx);
        int startIdx = startstops[idx];
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        if (n>0) {
            Virial virialSum(0, 0, 0, 0, 0, 0);
            int myIdxInAngle = angles_shr[shr_idx].type >> 29;
            int idSelf = angles_shr[shr_idx].ids[myIdxInAngle];

            int idxSelf = idToIdxs[idSelf];
            // XXX cast as double3 for double precision
            float3 pos = make_float3(xs[idxSelf]);
            //double3 pos = make_double3(xs[idxSelf]);
            //printf("pos %f %f %f\n", 
            //float3 pos = make_float3(float4FromIndex(xs, idxSelf));
            // XXX cast as double3 for double precision
            float3 forceSum = make_float3(0, 0, 0);
            //double3 forceSum = make_double3(0, 0, 0);
            for (int i=0; i<n; i++) {
             //   printf("ANGLE! %d\n", i);
                AngleGPU angle = angles_shr[shr_idx + i];
                uint32_t typeFull = angle.type;
                myIdxInAngle = typeFull >> 29;
                int type = static_cast<int>((typeFull << 3) >> 3);
                //HERE
                ANGLETYPE angleType = parameters[type];
                // XXX cast as double3 ....
                float3 positions[3];
                //double3 positions[3];
                positions[myIdxInAngle] = pos;
                int toGet[2];
                if (myIdxInAngle==0) {
                    toGet[0] = 1;
                    toGet[1] = 2;
                } else if (myIdxInAngle==1) {
                    toGet[0] = 0;
                    toGet[1] = 2;
                } else if (myIdxInAngle==2) {
                    toGet[0] = 0;
                    toGet[1] = 1;
                }
                for (int i=0; i<2; i++) {
                    int idxOther = idToIdxs[angle.ids[toGet[i]]];
                    // XXX cast as double
                    positions[toGet[i]] = make_float3(xs[idxOther]);
                    //positions[toGet[i]] = make_double3(xs[idxOther]);
                }
                for (int i=1; i<3; i++) {
                    positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
                }
                // XXX cast as double3
                float3 directors[2];
                //double3 directors[2];
                directors[0] = positions[0] - positions[1];
                directors[1] = positions[2] - positions[1];
             //   printf("position Xs %f %f %f\n", positions[0].x, positions[1].x, positions[2].x);
              //  printf("director Xs %f %f\n", directors[0].x, directors[1].x);
              //  // XXX cast these 2 as double for double precision
                float distSqrs[2];
                float dists[2];
                //double distSqrs[2];
                //double dists[2];
                for (int i=0; i<2; i++) {
                    distSqrs[i] = lengthSqr(directors[i]);
                    // XXX cast as sqrt() instead of sqrtf() for double
                    dists[i] = sqrtf(distSqrs[i]);
                    //dists[i] = sqrt(distSqrs[i]);
                }
                float c = dot(directors[0], directors[1]);

                //double c = dot(directors[0], directors[1]);
             //   printf("prenorm c is %f\n", c);
                float invDistProd = 1.0f / (dists[0]*dists[1]);

                //double invDistProd = 1.0f / (dists[0]*dists[1]);
              //  printf("inv dist is %f\n", invDistProd);
                c *= invDistProd;
              //  printf("c is %f\n", c);
                if (c>1) {
                    c=1;
                } else if (c<-1) {
                    c=-1;
                }
                // XXX cast as double, change sqrtf() to sqrt() for double
                 float s = sqrtf(1-c*c);
                //double s = sqrt(1-c*c);
                if (s < SMALL) {
                    s = SMALL;
                }
                s = 1.0f / s;
                // XXX cast as double for double precision, change acosf() to acos()
                float theta = acosf(c);
                //double theta = acos(c);
                if (COMPUTEVIRIALS) {
                    // XXX cast as double3 for double precision
                     float3 allForces[3];
                    //double3 allForces[3];
                    evaluator.forcesAll(angleType, theta, s, c, distSqrs, directors, invDistProd, allForces);
                    // XXX if we are doing double precision, we need to re-cast as single for computeVirial() call
                    //     -- to get back single precision, comment out next 6 lines, uncomment the two below them
                    //        (provided other changes back to single have been made accordingly)
                    //float3 tmp_0 = make_float3(allForces[0]);
                    //float3 dir_0 = make_float3(directors[0]);
                    //float3 tmp_2 = make_float3(allForces[2]);
                    //float3 dir_1 = make_float3(directors[1]);
                    //computeVirial(virialSum, tmp_0,dir_0);
                    //computeVirial(virialSum, tmp_2,dir_1);
                    computeVirial(virialSum, allForces[0], directors[0]);
                    computeVirial(virialSum, allForces[2], directors[1]);
              
                    forceSum += allForces[myIdxInAngle];
                } else {
                    forceSum += evaluator.force(angleType, theta, s, c, distSqrs, directors, invDistProd, myIdxInAngle);
                }


            }
            // XXX cast as double4, cast as make_double4() on global variable
            //double4 curForce = make_double4(forces[idxSelf]);
            float4 curForce = forces[idxSelf];
            curForce += forceSum;
            // XXX if in double, must cast make_float4(curForce); else, set equal to curForce
            //forces[idxSelf] = make_float4(curForce);
            forces[idxSelf] = curForce;
            if (COMPUTEVIRIALS) {
                virialSum *= 1.0f / 3.0f;
                virials[idx] += virialSum;
            }
        }
    }
}






template <class ANGLETYPE, class EVALUATOR>
__global__ void compute_energy_angle(int nAtoms, float4 *xs, float *perParticleEng, int *idToIdxs, AngleGPU *angles, int *startstops, BoundsGPU bounds, ANGLETYPE *parameters_arg, int nParameters, bool usingSharedMemForParams, EVALUATOR evaluator) {

    int idx = GETIDX();
    extern __shared__ char all_shr[];
    int idxBeginCopy = startstops[blockDim.x*blockIdx.x];
    int idxEndCopy = startstops[min(nAtoms, blockDim.x*(blockIdx.x+1))];
    AngleGPU *angles_shr = (AngleGPU *) all_shr;
    int sizeAngles = (idxEndCopy - idxBeginCopy) * sizeof(AngleGPU);
    copyToShared<AngleGPU>(angles + idxBeginCopy, angles_shr, idxEndCopy - idxBeginCopy);
    ANGLETYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (ANGLETYPE *) (all_shr + sizeAngles);
        copyToShared<ANGLETYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();
    if (idx < nAtoms) {
        //printf("going to compute %d\n", idx);
        int startIdx = startstops[idx];
        int endIdx = startstops[idx+1];
        //so start/end is the index within the entire bond list.
        //startIdx - idxBeginCopy gives my index in shared memory
        int shr_idx = startIdx - idxBeginCopy;
        int n = endIdx - startIdx;
        if (n>0) {
            int myIdxInAngle = angles_shr[shr_idx].type >> 29;
            int idSelf = angles_shr[shr_idx].ids[myIdxInAngle];

            int idxSelf = idToIdxs[idSelf];
            float3 pos = make_float3(xs[idxSelf]);
            //float3 pos = make_float3(float4FromIndex(xs, idxSelf));
            float engSum = 0;
            for (int i=0; i<n; i++) {
             //   printf("ANGLE! %d\n", i);
                AngleGPU angle = angles_shr[shr_idx + i];
                uint32_t typeFull = angle.type;
                myIdxInAngle = typeFull >> 29;
                int type = ((typeFull << 3) >> 3);
                ANGLETYPE angleType = parameters[type];
                float3 positions[3];
                positions[myIdxInAngle] = pos;
                int toGet[2];
                if (myIdxInAngle==0) {
                    toGet[0] = 1;
                    toGet[1] = 2;
                } else if (myIdxInAngle==1) {
                    toGet[0] = 0;
                    toGet[1] = 2;
                } else if (myIdxInAngle==2) {
                    toGet[0] = 0;
                    toGet[1] = 1;
                }
                for (int i=0; i<2; i++) {
                    int idxOther = idToIdxs[angle.ids[toGet[i]]];
                    positions[toGet[i]] = make_float3(xs[idxOther]);
                }
                for (int i=1; i<3; i++) {
                    positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
                }
                float3 directors[2];
                directors[0] = positions[0] - positions[1];
                directors[1] = positions[2] - positions[1];
             //   printf("position Xs %f %f %f\n", positions[0].x, positions[1].x, positions[2].x);
              //  printf("director Xs %f %f\n", directors[0].x, directors[1].x);
                float distSqrs[2];
                float dists[2];
                for (int i=0; i<2; i++) {
                    distSqrs[i] = lengthSqr(directors[i]);
                    dists[i] = sqrtf(distSqrs[i]);
                }
                float c = dot(directors[0], directors[1]);
             //   printf("prenorm c is %f\n", c);
                float invDistProd = 1.0f / (dists[0]*dists[1]);
              //  printf("inv dist is %f\n", invDistProd);
                c *= invDistProd;
              //  printf("c is %f\n", c);
                if (c>1) {
                    c=1;
                } else if (c<-1) {
                    c=-1;
                }
                float s = sqrtf(1-c*c);
                if (s < SMALL) {
                    s = SMALL;
                }
                s = 1.0f / s;
                float theta = acosf(c);
                engSum += evaluator.energy(angleType, theta, directors);

            }
            perParticleEng[idxSelf] += engSum;
        }
    }
}

