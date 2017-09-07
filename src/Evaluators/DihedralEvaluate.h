template <class DIHEDRALTYPE, class EVALUATOR, bool COMPUTEVIRIALS> //don't need DihedralGPU, are all DihedralGPU.  Worry about later 
__global__ void compute_force_dihedral(int nDihedrals, float4 *xs, float4 *fs, int *idToIdxs, DihedralGPU *dihedrals, BoundsGPU bounds, DIHEDRALTYPE *parameters_arg, int nParameters, Virial *virials, bool usingSharedMemForParams, EVALUATOR evaluator) {


    int idx = GETIDX();
    extern __shared__ char all_shr[];
    DIHEDRALTYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (DIHEDRALTYPE *) (all_shr);
        copyToShared<DIHEDRALTYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();
    if (idx < nDihedrals) {
        //      printf("going to compute %d\n", idx);
        Virial sumVirials(0, 0, 0, 0, 0, 0);
        int idxs[4];
        DihedralGPU dihedral = dihedrals[idx];

        uint32_t typeFull = dihedral.type;
        //b/c idx in forcer is stored in first three bytes
        int type = (typeFull << 3) >> 3;
        DIHEDRALTYPE dihedralType = parameters[type];

        float3 positions[4];


        for (int i=0; i<4; i++) {
            int idxOther = idToIdxs[dihedral.ids[i]];
            idxs[i] = idxOther;
            positions[i] = make_float3(xs[idxOther]);
        }
        for (int i=1; i<4; i++) {
            positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
        }
        float3 directors[3]; //vb_xyz in lammps
        float lenSqrs[3]; //bnmag2 in lammps
        float lens[3]; //bnmag in lammps
        float invLenSqrs[3]; //sb in lammps
        float invLens[3];
        directors[0] = positions[0] - positions[1];
        directors[1] = positions[2] - positions[1];
        directors[2] = positions[3] - positions[2];
        for (int i=0; i<3; i++) {
            //printf("directors %d is %f %f %f\n", i, directors[i].x, directors[i].y, directors[i].z);
            lenSqrs[i] = lengthSqr(directors[i]);
            lens[i] = sqrtf(lenSqrs[i]);
            invLenSqrs[i] = 1.0f / lenSqrs[i];
            invLens[i] = 1.0f / lens[i];
            //   printf("inv len sqrs %d is %f\n", i, invLenSqrs[i]);
        }


        float c0 = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
        //   printf("c0 is %f\n", c0);
        float c12Mags[2];
        float invMagProds[2]; //r12c1, 2 in lammps
        for (int i=0; i<2; i++) {
            float dotProd = dot(directors[i+1], directors[i]);
            if (i==1) {
                dotProd *= -1;
            }
            //      printf("ctmp is %f\n", dotProd);
            invMagProds[i] = invLens[i] * invLens[i+1];
            c12Mags[i] = dotProd * invMagProds[i]; //lammps variable names are opaque
            //      printf("c12 mag %d %f\n", i, c12Mags[i]);
        }

        float scValues[3]; //???, is s1, s2, s12 in lammps
        for (int i=0; i<2; i++) {
            float x = max(1 - c12Mags[i]*c12Mags[i], 0.0f);
            float sqrtVal = max(sqrtf(x), EPSILON);
            scValues[i] = 1.0 / sqrtVal;
        }
        scValues[2] = scValues[0] * scValues[1];


        for (int i=0; i<2; i++) {
            scValues[i] *= scValues[i]; 
        }
        //   printf("sc values %f %f %f\n", scValues[0], scValues[1], scValues[2]);
        float c = (c0 + c12Mags[0]*c12Mags[1]) * scValues[2];

        float3 cVector;
        cVector.x = directors[0].y*directors[1].z - directors[0].z*directors[1].y;
        cVector.y = directors[0].z*directors[1].x - directors[0].x*directors[1].z;
        cVector.z = directors[0].x*directors[1].y - directors[0].y*directors[1].x;
        float cVectorLen = length(cVector);
        float dx = dot(cVector, directors[2]) * invLens[2] / cVectorLen;
        //printf("c xyz %f %f %f directors xyz %f %f %f\n", cVector.x, cVector.y, cVector.z, directors[2].x, directors[2].y, directors[2].z);
        //printf("c is %f\n", c);
        if (c > 1.0f) {
            c = 1.0f;
        } else if (c < -1.0f) {
            c = -1.0f;
        }
        float phi = acosf(c);
        // printf("phi is %f\n", phi);
        // printf("dx is %f\n", dx);
        if (dx < 0) {
            phi = -phi;
        }
        // printf("phi is %f\n", phi);

        //printf("no force\n");
        float dPotential = -1.0f * evaluator.dPotential(dihedralType, phi);
        float sinPhi = sinf(phi);
        float absSinPhi = sinPhi < 0 ? -sinPhi : sinPhi;
        if (absSinPhi < EPSILON) {
            sinPhi = EPSILON;
        }
        dPotential /= sinPhi;
        float3 forces[4];


        c *= dPotential;
        scValues[2] *= dPotential;
        float a11 = c * invLenSqrs[0] * scValues[0];
        float a22 = -invLenSqrs[1] * (2.0f*c0*scValues[2] - c*(scValues[0]+scValues[1]));
        float a33 = c*invLenSqrs[2]*scValues[1];
        float a12 = -invMagProds[0] * (c12Mags[0] * c * scValues[0] + c12Mags[1] * scValues[2]);
        float a13 = -invLens[0] * invLens[2] * scValues[2];
        float a23 = invMagProds[1] * (c12Mags[1]*c*scValues[1] + c12Mags[0]*scValues[2]);
        float3 sFloat3 = make_float3(
                                     a12*directors[0].x + a22*directors[1].x + a23*directors[2].x
                                     ,  a12*directors[0].y + a22*directors[1].y + a23*directors[2].y
                                     ,  a12*directors[0].z + a22*directors[1].z + a23*directors[2].z
                                    );
        //printf("ssomething valyes %f %f %f\n", sFloat3.x, sFloat3.y, sFloat3.z);
        //printf("comps %f %f %f %f %f %f\n", a12, directors[0].x,  a22, directors[1].x,  a23, directors[2].x);
        float3 a11Dir1 = directors[0] * a11;
        float3 a12Dir2 = directors[1] * a12;
        float3 a13Dir3 = directors[2] * a13;
        forces[0].x = a11Dir1.x + a12Dir2.x + a13Dir3.x;
        forces[0].y = a11Dir1.y + a12Dir2.y + a13Dir3.y;
        forces[0].z = a11Dir1.z + a12Dir2.z + a13Dir3.z;
        forces[1] = -sFloat3 - forces[0];


        float3 a13Dir1 = directors[0] * a13;
        float3 a23Dir2 = directors[1] * a23;
        float3 a33Dir3 = directors[2] * a33;
        forces[3].x = a13Dir1.x + a23Dir2.x + a33Dir3.x;
        forces[3].y = a13Dir1.y + a23Dir2.y + a33Dir3.y;
        forces[3].z = a13Dir1.z + a23Dir2.z + a33Dir3.z;
        forces[2] = sFloat3 - forces[3];
        //printf("phi is %f\n", phi);
        for (int i=0; i<4; i++) {
            atomicAdd(&(fs[idxs[i]].x), (forces[i].x));
            atomicAdd(&(fs[idxs[i]].y), (forces[i].y));
            atomicAdd(&(fs[idxs[i]].z), (forces[i].z));
        //    printf("f %d is %f %f %f\n", i, forces[i].x, forces[i].y, forces[i].z);
        }




        if (COMPUTEVIRIALS) {
            computeVirial(sumVirials, forces[0], directors[0]);
            computeVirial(sumVirials, forces[2], directors[1]);
            computeVirial(sumVirials, forces[3], directors[1] + directors[2]);
            //just adding virials to one of them
            for (int i=0; i<6; i++) {
                //printf("virial %d %f\n", i, sumVirials[i]);
                atomicAdd(&(virials[idxs[0]][i]), sumVirials[i]);
            }
        }
    }
}


template <class DIHEDRALTYPE, class EVALUATOR>
__global__ void compute_energy_dihedral(int nDihedrals, float4 *xs, float *perParticleEng, int *idToIdxs, DihedralGPU *dihedrals, BoundsGPU bounds, DIHEDRALTYPE *parameters_arg, int nParameters, bool usingSharedMemForParams, EVALUATOR evaluator) {
 
    int idx = GETIDX();
    extern __shared__ char all_shr[];
    DIHEDRALTYPE *parameters;
    if (usingSharedMemForParams) {
        parameters = (DIHEDRALTYPE *) (all_shr);
        copyToShared<DIHEDRALTYPE>(parameters_arg, parameters, nParameters);
    } else {
        parameters = parameters_arg;
    }
    __syncthreads();
    if (idx < nDihedrals) {
        //      printf("going to compute %d\n", idx);
        Virial sumVirials(0, 0, 0, 0, 0, 0);
        int idxs[4];
        DihedralGPU dihedral = dihedrals[idx];

        uint32_t typeFull = dihedral.type;
        //b/c idx in forcer is stored in first three bytes
        int type = (typeFull << 3) >> 3;
        DIHEDRALTYPE dihedralType = parameters[type];

        float3 positions[4];


        for (int i=0; i<4; i++) {
            int idxOther = idToIdxs[dihedral.ids[i]];
            idxs[i] = idxOther;
            positions[i] = make_float3(xs[idxOther]);
        }
        for (int i=1; i<4; i++) {
            positions[i] = positions[0] + bounds.minImage(positions[i]-positions[0]);
        }
        float3 directors[3]; //vb_xyz in lammps
        float lenSqrs[3]; //bnmag2 in lammps
        float lens[3]; //bnmag in lammps
        float invLens[3];
        directors[0] = positions[0] - positions[1];
        directors[1] = positions[2] - positions[1];
        directors[2] = positions[3] - positions[2];
        for (int i=0; i<3; i++) {
            //printf("directors %d is %f %f %f\n", i, directors[i].x, directors[i].y, directors[i].z);
            lenSqrs[i] = lengthSqr(directors[i]);
            lens[i] = sqrtf(lenSqrs[i]);
            invLens[i] = 1.0f / lens[i];
        }


        float c0 = dot(directors[0], directors[2]) * invLens[0] * invLens[2];
        //   printf("c0 is %f\n", c0);
        float c12Mags[2];
        float invMagProds[2]; //r12c1, 2 in lammps
        for (int i=0; i<2; i++) {
            float dotProd = dot(directors[i+1], directors[i]);
            if (i==1) {
                dotProd *= -1;
            }
            //      printf("ctmp is %f\n", dotProd);
            invMagProds[i] = invLens[i] * invLens[i+1];
            c12Mags[i] = dotProd * invMagProds[i]; //lammps variable names are opaque
            //      printf("c12 mag %d %f\n", i, c12Mags[i]);
        }

        float scValues[3]; //???, is s1, s2, s12 in lammps
        for (int i=0; i<2; i++) {
            float x = max(1 - c12Mags[i]*c12Mags[i], 0.0f);
            float sqrtVal = max(sqrtf(x), EPSILON);
            scValues[i] = 1.0 / sqrtVal;
        }
        scValues[2] = scValues[0] * scValues[1];


        for (int i=0; i<2; i++) {
            scValues[i] *= scValues[i]; 
        }
        //   printf("sc values %f %f %f\n", scValues[0], scValues[1], scValues[2]);
        float c = (c0 + c12Mags[0]*c12Mags[1]) * scValues[2];

        float3 cVector;
        cVector.x = directors[0].y*directors[1].z - directors[0].z*directors[1].y;
        cVector.y = directors[0].z*directors[1].x - directors[0].x*directors[1].z;
        cVector.z = directors[0].x*directors[1].y - directors[0].y*directors[1].x;
        float cVectorLen = length(cVector);
        float dx = dot(cVector, directors[2]) * invLens[2] / cVectorLen;
        //printf("c xyz %f %f %f directors xyz %f %f %f\n", cVector.x, cVector.y, cVector.z, directors[2].x, directors[2].y, directors[2].z);
        //printf("c is %f\n", c);
        if (c > 1.0f) {
            c = 1.0f;
        } else if (c < -1.0f) {
            c = -1.0f;
        }
        float phi = acosf(c);
        // printf("phi is %f\n", phi);
        // printf("dx is %f\n", dx);
        if (dx < 0) {
            phi = -phi;
        }
        // printf("phi is %f\n", phi);

        //printf("no force\n");
        float potential = evaluator.potential(dihedralType, phi) * 0.25f;
        for (int i=0; i<4; i++) {
            atomicAdd(perParticleEng + idxs[i], potential);
        }
    }
}
