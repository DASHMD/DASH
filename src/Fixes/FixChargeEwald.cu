#include "FixChargeEwald.h"

#include "BoundsGPU.h"
#include "cutils_func.h"
#include "cutils_math.h"
#include "GridGPU.h"
#include "State.h"
#include <cufft.h>
#include "globalDefs.h"
#include <fstream>
#include "Virial.h"
#include "helpers.h"

#include "PairEvaluatorNone.h"
#include "EvaluatorWrapper.h"
// #include <cmath>
using namespace std;
namespace py = boost::python;
const std::string chargeEwaldType = "ChargeEwald";

// #define THREADS_PER_BLOCK_

// MW: Note that this function is a verbatim copy of that which appears in GridGPU.cu
//     consider combining
__global__ void computeCentroid(float4 *centroids, float4 *xs, int nAtoms, int nPerRingPoly, BoundsGPU bounds) {
   int idx = GETIDX();
    int nRingPoly = nAtoms / nPerRingPoly;
    if (idx < nRingPoly) {
        int baseIdx = idx * nPerRingPoly;
        float3 init = make_float3(xs[baseIdx]);
        float3 diffSum = make_float3(0, 0, 0);
        for (int i=baseIdx+1; i<baseIdx + nPerRingPoly; i++) {
            float3 next = make_float3(xs[i]);
            float3 dx = bounds.minImage(next - init);
            diffSum += dx;
        }
        diffSum /= nPerRingPoly;
        float3 unwrappedPos = init + diffSum;
        float3 trace = bounds.trace();
        float3 diffFromLo = unwrappedPos - bounds.lo;
        float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
        float3 wrappedPos = unwrappedPos - trace * imgs * bounds.periodic;

        centroids[idx] = make_float4(wrappedPos);
    }

}

// MW: This is a duplicated function from GridGPU.cu
 __global__ void periodicWrapCpy(float4 *xs, int nAtoms, BoundsGPU bounds) {
 
     int idx = GETIDX(); 
     if (idx < nAtoms) {
         
         float4 pos = xs[idx];
         
         float id = pos.w;
         float3 trace = bounds.trace();
         float3 diffFromLo = make_float3(pos) - bounds.lo;
         float3 imgs = floorf(diffFromLo / trace); //are unskewed at this point
         pos -= make_float4(trace * imgs * bounds.periodic);
         pos.w = id;
         //if (not(pos.x==orig.x and pos.y==orig.y and pos.z==orig.z)) { //sigh
         if (imgs.x != 0 or imgs.y != 0 or imgs.z != 0) {
             xs[idx] = pos;
         }
     }
 
 }
//different implementation for different interpolation orders
//TODO template
//order 1 nearest point
__global__ void map_charge_to_grid_order_1_cu(int nRingPoly, int nPerRingPoly, float4 *xs,  float *qs,  BoundsGPU bounds,
                                      int3 sz,float *grid/*convert to float for cufffComplex*/,float  Qunit) {

    int idx = GETIDX();
    if (idx < nRingPoly) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole)-bounds.lo;

        float qi = Qunit*qs[idx * nPerRingPoly];
        
        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        //or
        int3 p=nearest_grid_point;
        if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
        if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
        if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
        if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
        if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
        if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
        atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], 1.0*qi);
    }
}

inline __host__ __device__ float W_p_3(int i,float x){
    if (i==-1) return 0.125-0.5*x+0.5*x*x;
    if (i== 0) return 0.75-x*x;
    /*if (i== 1)*/ return 0.125+0.5*x+0.5*x*x;
}


__global__ void map_charge_to_grid_order_3_cu(int nRingPoly, int nPerRingPoly, float4 *xs,  float *qs,  BoundsGPU bounds,
                                      int3 sz,float *grid/*convert to float for cufffComplex*/,float  Qunit) {

    int idx = GETIDX();
    if (idx < nRingPoly) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole)-bounds.lo;

        float qi = Qunit*qs[idx * nPerRingPoly];
        
        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        float3 d=pos/h-make_float3(nearest_grid_point);
        
        int3 p=nearest_grid_point;
        for (int ix=-1;ix<=1;ix++){
          p.x=nearest_grid_point.x+ix;
          float charge_yz_w=qi*W_p_3(ix,d.x);
          for (int iy=-1;iy<=1;iy++){
            p.y=nearest_grid_point.y+iy;
            float charge_z_w=charge_yz_w*W_p_3(iy,d.y);
            for (int iz=-1;iz<=1;iz++){
                p.z=nearest_grid_point.z+iz;
                float charge_w=charge_z_w*W_p_3(iz,d.z);
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                if ((p.x<0) or (p.x>sz.x-1)) printf("grid point miss x  %d, %d, %d, %f \n", idx,p.x,nearest_grid_point.x,pos.x);
                if ((p.y<0) or (p.y>sz.y-1)) printf("grid point miss y  %d, %d, %d, %f \n", idx,p.y,nearest_grid_point.y,pos.y);
                if ((p.z<0) or (p.z>sz.z-1)) printf("grid point miss z  %d, %d, %d, %f \n", idx,p.z,nearest_grid_point.z,pos.z);
                
                atomicAdd(&grid[p.x*sz.y*sz.z*2+p.y*sz.z*2+p.z*2], charge_w);
                
            }
          }
        }
    }
}


__global__ void map_charge_set_to_zero_cu(int3 sz,cufftComplex *grid) {
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z))                  
         grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]=make_cuComplex (0.0f, 0.0f);    
}

__device__ float sinc(float x){
  if ((x<0.1)&&(x>-0.1)){
    float x2=x*x;
    return 1.0 - x2*0.16666666667f + x2*x2*0.008333333333333333f - x2*x2*x2*0.00019841269841269841f;    
  }
    else return sin(x)/x;
}

__global__ void Green_function_cu(BoundsGPU bounds, int3 sz,float *Green_function,float alpha,
                                  //now some parameter for Gf calc
                                  int sum_limits, int intrpl_order) {
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
          float3 h =bounds.trace()/make_float3(sz);
          
          //         2*PI
          float3 k= 6.28318530717958647693f*make_float3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;
          

          //OK GF(k)  = 4Pi/K^2 [SumforM(W(K+M)^2  exp(-(K+M)^2/4alpha) dot(K,K+M)/(K+M^2))] / 
          //                    [SumforM^2(W(K+M)^2)]
             
             
          float sum1=0.0f;   
          float sum2=0.0f;   
          float k2=lengthSqr(k);
          float Fouralpha2inv=0.25/alpha/alpha;
          if (k2!=0.0){
              for (int ix=-sum_limits;ix<=sum_limits;ix++){//TODO different limits 
                for (int iy=-sum_limits;iy<=sum_limits;iy++){
                  for (int iz=-sum_limits;iz<=sum_limits;iz++){
                      float3 kpM=k+6.28318530717958647693f*make_float3(ix,iy,iz)/h;
//                             kpM.x+=6.28318530717958647693f/h.x*ix;//TODO rewrite
//                             kpM.y+=6.28318530717958647693f/h.y*iy;
//                             kpM.z+=6.28318530717958647693f/h.z*iz;
                            float kpMlen=lengthSqr(kpM);
                            float W=sinc(kpM.x*h.x*0.5)*sinc(kpM.y*h.y*0.5)*sinc(kpM.z*h.z*0.5);
//                             for(int p=1;p<intrpl_order;p++)
//                                   W*=W;
    //                          W*=h;//not need- cancels out
//                             float W2=W*W;
                             float W2=pow(W,intrpl_order*2);
                            //4*PI
                            sum1+=12.56637061435917295385*exp(-kpMlen*Fouralpha2inv)*dot(k,kpM)/kpMlen*W2;
                            sum2+=W2;
                  }
                }
              }
              Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z]=sum1/(sum2*sum2)/k2;
          }else{
              Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z]=0.0f;
          }
      }
             
}

__global__ void potential_cu(int3 sz,float *Green_function,
                                    cufftComplex *FFT_qs, cufftComplex *FFT_phi){
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
        FFT_phi[id.x*sz.y*sz.z+id.y*sz.z+id.z]=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z]*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
//TODO after Inverse FFT divide by volume
      }
}

__global__ void E_field_cu(BoundsGPU bounds, int3 sz,float *Green_function, cufftComplex *FFT_qs,
                           cufftComplex *FFT_Ex,cufftComplex *FFT_Ey,cufftComplex *FFT_Ez){
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
          //K vector
          float3 k= 6.28318530717958647693f*make_float3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;        
          
          //ik*q(k)*Gf(k)
          cufftComplex Ex,Ey,Ez;
          float GF=Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          cufftComplex q=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];

          Ex.y= k.x*q.x*GF;
          Ex.x=-k.x*q.y*GF;
          Ey.y= k.y*q.x*GF;
          Ey.x=-k.y*q.y*GF;
          Ez.y= k.z*q.x*GF;
          Ez.x=-k.z*q.y*GF;
          
          FFT_Ex[id.x*sz.y*sz.z+id.y*sz.z+id.z]=Ex;
          FFT_Ey[id.x*sz.y*sz.z+id.y*sz.z+id.z]=Ey;
          FFT_Ez[id.x*sz.y*sz.z+id.y*sz.z+id.z]=Ez;
          //TODO after Inverse FFT divide by -volume
      }
}


__global__ void Ewald_long_range_forces_order_1_cu(int nRingPoly, int nPerRingPoly, float4 *xs, float4 *fs, 
                                                   float *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez,float  Qunit,
                                                   bool storeForces, uint *ids, float4 *storedForces) {
    int idx = GETIDX();
    if (idx < nRingPoly) {
        float4 posWhole= xs[idx];
        float3 pos     = make_float3(posWhole)-bounds.lo;
        int    baseIdx = idx*nPerRingPoly;
        float  qi      = qs[baseIdx];
        
        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);

        int3 p=nearest_grid_point;        
        if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
        if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
        if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
        if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
        if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
        if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
        
        //get E field
        float3 E;
        float volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;
        E.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        E.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        E.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
        
        // Apply force on centroid to all time slices for given atom
        float3 force= Qunit*qi*E;
        for (int i = 0; i< nPerRingPoly; i++) {
            fs[baseIdx + i] += force; 
        }

        if (storeForces) {
            for (int i = 0; i < nPerRingPoly; i++) {
                storedForces[ids[baseIdx+i]] = make_float4(force.x, force.y, force.z, 0);
            }
        }
    }
}


__global__ void Ewald_long_range_forces_order_3_cu(int nRingPoly, int nPerRingPoly, float4 *xs, float4 *fs, 
                                                   float *qs, BoundsGPU bounds,
                                                   int3 sz, cufftComplex *FFT_Ex,
                                                    cufftComplex *FFT_Ey,cufftComplex *FFT_Ez,float  Qunit,
                                                   bool storeForces, uint *ids, float4 *storedForces) {
    int idx = GETIDX();
    if (idx < nRingPoly) {
        float4 posWhole= xs[idx];
        float3 pos     = make_float3(posWhole)-bounds.lo;
        int    baseIdx = idx*nPerRingPoly;
        float  qi      = qs[baseIdx];

        //find nearest grid point
        float3 h=bounds.trace()/make_float3(sz);
        int3 nearest_grid_point=make_int3((pos+0.5*h)/h);
        
        //distance from nearest_grid_point /h
        float3 d=pos/h-make_float3(nearest_grid_point);

        float3 E=make_float3(0,0,0);
        float volume=bounds.trace().x*bounds.trace().y*bounds.trace().z;

        int3 p=nearest_grid_point;
        for (int ix=-1;ix<=1;ix++){
          p.x=nearest_grid_point.x+ix;
          for (int iy=-1;iy<=1;iy++){
            p.y=nearest_grid_point.y+iy;
            for (int iz=-1;iz<=1;iz++){
                p.z=nearest_grid_point.z+iz;
                if (p.x>0) p.x-=int(p.x/sz.x)*sz.x;
                if (p.y>0) p.y-=int(p.y/sz.y)*sz.y;
                if (p.z>0) p.z-=int(p.z/sz.z)*sz.z;
                if (p.x<0) p.x-=int((p.x+1)/sz.x-1)*sz.x;
                if (p.y<0) p.y-=int((p.y+1)/sz.y-1)*sz.y;
                if (p.z<0) p.z-=int((p.z+1)/sz.z-1)*sz.z;
                float3 Ep;
                float W_xyz=W_p_3(ix,d.x)*W_p_3(iy,d.y)*W_p_3(iz,d.z);
                
                Ep.x= -FFT_Ex[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.y= -FFT_Ey[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                Ep.z= -FFT_Ez[p.x*sz.y*sz.z+p.y*sz.z+p.z].x/volume;
                E+=W_xyz*Ep;
            }
          }
        }
               
        float3 force= Qunit*qi*E;
        // Apply force on centroid to all time slices for given atom
        for (int i = 0; i < nPerRingPoly; i++) {
            fs[baseIdx + i] += force;
        }

        if (storeForces) {
            for (int i = 0; i < nPerRingPoly; i++) {
                storedForces[ids[baseIdx+i]] = make_float4(force.x, force.y, force.z, 0);
            }
        }
    }
}


__global__ void Energy_cu(int3 sz,float *Green_function,
                                    cufftComplex *FFT_qs, cufftComplex *E_grid){
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
        cufftComplex qi=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];
        E_grid[id.x*sz.y*sz.z+id.y*sz.z+id.z]
            =make_cuComplex((qi.x*qi.x+qi.y*qi.y)*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z],0.0);
//TODO after Inverse FFT divide by volume
      }
}


__global__ void virials_cu(BoundsGPU bounds,int3 sz,Virial *dest,float alpha, float *Green_function,cufftComplex *FFT_qs,int warpSize){
      int3 id = make_int3( blockIdx.x*blockDim.x + threadIdx.x,
                          blockIdx.y*blockDim.y + threadIdx.y,
                          blockIdx.z*blockDim.z + threadIdx.z);

      if ((id.x<sz.x)&&(id.y<sz.y)&&(id.z<sz.z)){
          float3 k= 6.28318530717958647693f*make_float3(id)/bounds.trace();
          if (id.x>sz.x/2) k.x= 6.28318530717958647693f*(id.x-sz.x)/bounds.trace().x;
          if (id.y>sz.y/2) k.y= 6.28318530717958647693f*(id.y-sz.y)/bounds.trace().y;
          if (id.z>sz.z/2) k.z= 6.28318530717958647693f*(id.z-sz.z)/bounds.trace().z;        
          float klen=lengthSqr(k);
          cufftComplex qi=FFT_qs[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          float E=(qi.x*qi.x+qi.y*qi.y)*Green_function[id.x*sz.y*sz.z+id.y*sz.z+id.z];
          
          float differential=-2.0*(1.0/klen+0.25/(alpha*alpha));
          if (klen==0.0) {differential=0.0;E=0.0;}
          
          Virial virialstmp = Virial(0, 0, 0, 0, 0, 0);   
          virialstmp[0]=(1.0+differential*k.x*k.x)*E; //xx
          virialstmp[1]=(1.0+differential*k.y*k.y)*E; //yy
          virialstmp[2]=(1.0+differential*k.z*k.z)*E; //zz
          virialstmp[3]=(differential*k.x*k.y)*E; //xy
          virialstmp[4]=(differential*k.x*k.z)*E; //xz
          virialstmp[5]=(differential*k.y*k.z)*E; //yz

//           virials[id.x*sz.y*sz.z+id.y*sz.z+id.z]=virialstmp;
//           __syncthreads();
          extern __shared__ Virial tmpV[]; 
//           const int copyBaseIdx = blockDim.x*blockIdx.x * N_DATA_PER_THREAD + threadIdx.x;
//           const int copyIncrement = blockDim.x;
          tmpV[threadIdx.x*blockDim.y*blockDim.z+threadIdx.y*blockDim.z+threadIdx.z]=virialstmp;
          int curLookahead=1;
          int numLookaheadSteps = log2f(blockDim.x*blockDim.y*blockDim.z-1);
          const int sumBaseIdx = threadIdx.x*blockDim.y*blockDim.z+threadIdx.y*blockDim.z+threadIdx.z;
          __syncthreads();
          for (int i=0; i<=numLookaheadSteps; i++) {
              if (! (sumBaseIdx % (curLookahead*2))) {
                  tmpV[sumBaseIdx] += tmpV[sumBaseIdx + curLookahead];
              }
              curLookahead *= 2;
//               if (curLookahead >= (warpSize)) {//Doesn't work in 3D case 
                  __syncthreads();
//               }
          } 

          if (sumBaseIdx  == 0) {
            
              atomicAdd(&(dest[0].vals[0]), tmpV[0][0]);
              atomicAdd(&(dest[0].vals[1]), tmpV[0][1]);
              atomicAdd(&(dest[0].vals[2]), tmpV[0][2]);
              atomicAdd(&(dest[0].vals[3]), tmpV[0][3]);
              atomicAdd(&(dest[0].vals[4]), tmpV[0][4]);
              atomicAdd(&(dest[0].vals[5]), tmpV[0][5]);
          }          
      }
      
}


#define N_DATA_PER_THREAD 4 //just taken from cutils_func.h 
__global__ void sum_virials_cu(Virial *dest, Virial *src, int n, int warpSize){
      extern __shared__ Virial tmpV[]; 
    const int copyBaseIdx = blockDim.x*blockIdx.x * N_DATA_PER_THREAD + threadIdx.x;
    const int copyIncrement = blockDim.x;
    for (int i=0; i<N_DATA_PER_THREAD; i++) {
        int step = i * copyIncrement;
        if (copyBaseIdx + step < n) {
            tmpV[threadIdx.x + step] = src[copyBaseIdx + step];
            
        } else {
            tmpV[threadIdx.x + step] =Virial(0, 0, 0, 0, 0, 0);
        }
    }
    int curLookahead = N_DATA_PER_THREAD;
    int numLookaheadSteps = log2f(blockDim.x-1);
    const int sumBaseIdx = threadIdx.x * N_DATA_PER_THREAD;
    __syncthreads();
    for (int i=sumBaseIdx+1; i<sumBaseIdx + N_DATA_PER_THREAD; i++) {
        tmpV[sumBaseIdx] += tmpV[i];
    }
    for (int i=0; i<=numLookaheadSteps; i++) {
        if (! (sumBaseIdx % (curLookahead*2))) {
            tmpV[sumBaseIdx] += tmpV[sumBaseIdx + curLookahead];
        }
        curLookahead *= 2;
        if (curLookahead >= (N_DATA_PER_THREAD * warpSize)) {
            __syncthreads();
        }
    }
    if (threadIdx.x  == 0) {
        atomicAdd(&(dest[0].vals[0]), tmpV[0][0]);
        atomicAdd(&(dest[0].vals[1]), tmpV[0][1]);
        atomicAdd(&(dest[0].vals[2]), tmpV[0][2]);
        atomicAdd(&(dest[0].vals[3]), tmpV[0][3]);
        atomicAdd(&(dest[0].vals[4]), tmpV[0][4]);
        atomicAdd(&(dest[0].vals[5]), tmpV[0][5]);
    }
}
/*
template < bool COMPUTE_VIRIALS>
__global__ void compute_short_range_forces_cu(int nAtoms, float4 *xs, float4 *fs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, float *qs, float alpha, float rCut, BoundsGPU bounds, int warpSize, float onetwoStr, float onethreeStr, float onefourStr, Virial *__restrict__ virials, Virial *virialField, float volume,float  conversion) {

    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
 //   printf("USING SHORT RANGE FORCES IN VIRIAL.  THIS KERNEL IS INCORRECT\n");
    Virial virialsSum = Virial(0, 0, 0, 0, 0, 0);   
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole);

        float3 forceSum = make_float3(0, 0, 0);
        float qi = qs[idx];

        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            uint otherIdx = otherIdxRaw & EXCL_MASK;
            float3 otherPos = make_float3(xs[otherIdx]);
            //then wrap and compute forces!
            float3 dr = bounds.minImage(pos - otherPos);
            float lenSqr = lengthSqr(dr);
            //   printf("dist is %f %f %f\n", dr.x, dr.y, dr.z);
            if (lenSqr < rCut*rCut) {
                float multiplier = multipliers[neighDist];
                if (multiplier) {
                    float len=sqrtf(lenSqr);
                    float qj = qs[otherIdx];

                    float r2inv = 1.0f/lenSqr;
                    float rinv = 1.0f/len;                                   //1/Sqrt(Pi)
                    float forceScalar = conversion*qi*qj*(erfcf((alpha*len))*rinv+(2.0*0.5641895835477563*alpha)*exp(-alpha*alpha*lenSqr))*r2inv* multiplier;

                    
                    float3 forceVec = dr * forceScalar;
                    forceSum += forceVec;
//                     if ((::isnan(forceScalar)) or (abs(forceScalar)>1E6))  printf("short ewald nan %f ,%d ,%d %f \n", forceScalar,idx, otherIdx,pos.x);  
                    if (COMPUTE_VIRIALS) {
                        computeVirial(virialsSum, forceVec, dr);
                    
                    }
                }
            }

        }   
        fs[idx] += forceSum; //operator for float4 + float3
        if (COMPUTE_VIRIALS) {
            //printf("vir %f %f %f %f %f %f\n", virialsSum.vals[0], virialsSum.vals[1], virialsSum.vals[2], virial_per_particle.vals[0],virial_per_particle.vals[1],virial_per_particle.vals[2]);
            Virial field = virialField[0];
            field /= (nAtoms * volume);
            virialsSum+=field;
            virials[idx] += virialsSum;
        }
    }

}
*/
/*
__global__ void compute_short_range_energies_cu(int nAtoms, float4 *xs, uint16_t *neighborCounts, uint *neighborlist, uint32_t *cumulSumMaxPerBlock, float *qs, float alpha, float rCut, BoundsGPU bounds, int warpSize, float onetwoStr, float onethreeStr, float onefourStr,float *perParticleEng, float field_energy_per_particle,float  conversion) {

    float multipliers[4] = {1, onetwoStr, onethreeStr, onefourStr};
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 posWhole = xs[idx];
        float3 pos = make_float3(posWhole);

        float EngSum = 0.0f;
        float qi = qs[idx];

        int baseIdx = baseNeighlistIdx(cumulSumMaxPerBlock, warpSize);
        int numNeigh = neighborCounts[idx];
        for (int i=0; i<numNeigh; i++) {
            int nlistIdx = baseIdx + warpSize * i;
            uint otherIdxRaw = neighborlist[nlistIdx];
            uint neighDist = otherIdxRaw >> 30;
            uint otherIdx = otherIdxRaw & EXCL_MASK;
            float3 otherPos = make_float3(xs[otherIdx]);
            //then wrap and compute forces!
            float3 dr = bounds.minImage(pos - otherPos);
            float lenSqr = lengthSqr(dr);
            //   printf("dist is %f %f %f\n", dr.x, dr.y, dr.z);
            if (lenSqr < rCut*rCut) {
                float multiplier = multipliers[neighDist];
                if (multiplier) {
                    float len=sqrtf(lenSqr);
                    float qj = qs[otherIdx];

//                     float r2inv = 1.0f/lenSqr;
                    float rinv = 1.0f/len;                 
                    float eng = conversion*0.5*qi*qj*(erfcf((alpha*len))*rinv)*multiplier;
                    
                    EngSum += eng;
   
                }
            }

        }   
        perParticleEng[idx] += EngSum+field_energy_per_particle; 

    }

}
*/
__global__ void applyStoredForces(int  nAtoms,
                float4 *fs,
                uint *ids, float4 *fsStored) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        float4 cur = fs[idx];
        float3 stored = make_float3(fsStored[ids[idx]]);
        cur += stored;
        fs[idx] = cur;
    }
}
__global__ void mapVirialToSingleAtom(Virial *atomVirials, Virial *fieldVirial, float volume) {
    //just mapping to one atom for now.  If we're looking at per-atom properties, should change to mapping to all atoms evenly
    atomVirials[0][threadIdx.x] += 0.5 * fieldVirial[0][threadIdx.x] / volume;
}


__global__ void mapEngToParticles(int nAtoms, float eng, float *engs) {
    int idx = GETIDX();
    if (idx < nAtoms) {
        engs[idx] += eng;
    }
}

FixChargeEwald::FixChargeEwald(SHARED(State) state_, string handle_, string groupHandle_): FixCharge(state_, handle_, groupHandle_, chargeEwaldType, true){
    cufftCreate(&plan);
    canOffloadChargePairCalc = true;
    modeIsError = false;
    sz = make_int3(32, 32, 32);
    malloced = false;
    longRangeInterval = 1;
    setEvalWrapper();
}


FixChargeEwald::~FixChargeEwald(){
    cufftDestroy(plan);
    if (malloced) {
        cudaFree(FFT_Qs);
        cudaFree(FFT_Ex);
        cudaFree(FFT_Ey);
        cudaFree(FFT_Ez);
    }
}


//Root mean square force error estimation
const double amp_table[][7] = {
        {2.0/3.0,           0,                 0,                    0,                        0,                         0,                                0},
        {1.0/50.0,          5.0/294.0,         0,                    0,                        0,                         0,                                0},
        {1.0/588.0,         7.0/1440.0,        21.0/3872.0,          0,                        0,                         0,                                0},
        {1.0/4320.0,        3.0/1936.0,        7601.0/2271360.0,     143.0/28800.0,            0,                         0,                                0},
        {1.0/23232.0,       7601.0/12628160.0, 143.0/69120.0,        517231.0/106536960.0,     106640677.0/11737571328.0, 0,                                0},
        {691.0/68140800.0,  13.0/57600.0,      47021.0/35512320.0,   9694607.0/2095994880.0,   733191589.0/59609088000.0, 326190917.0/11700633600.0,        0},
        {1.0/345600.0,      3617.0/35512320.0, 745739.0/838397952.0, 56399353.0/12773376000.0, 25091609.0/1560084480.0,   1755948832039.0/36229939200000.0, 48887769399.0/37838389248.0}
}; 


double FixChargeEwald :: DeltaF_k(double t_alpha){
    int nAtoms = state->atoms.size(); 
   double sumx=0.0,sumy=0.0,sumz=0.0;
   for( int m=0;m<interpolation_order;m++){
       double amp=amp_table[interpolation_order-1][m];
       sumx+=amp*pow(h.x*t_alpha,2*m);
       sumy+=amp*pow(h.y*t_alpha,2*m);
       sumz+=amp*pow(h.z*t_alpha,2*m);
   }
   return total_Q2/3.0*(1.0/(L.x*L.x)*pow(t_alpha*h.x,interpolation_order)*sqrt(t_alpha*L.x/nAtoms*sqrt(2.0*M_PI)*sumx)+
                        1.0/(L.y*L.y)*pow(t_alpha*h.y,interpolation_order)*sqrt(t_alpha*L.y/nAtoms*sqrt(2.0*M_PI)*sumy)+
                        1.0/(L.z*L.z)*pow(t_alpha*h.z,interpolation_order)*sqrt(t_alpha*L.z/nAtoms*sqrt(2.0*M_PI)*sumz));
 }
 
double  FixChargeEwald :: DeltaF_real(double t_alpha){  
    int nAtoms = state->atoms.size(); 
   return 2*total_Q2/sqrt(nAtoms*r_cut*L.x*L.y*L.z)*exp(-t_alpha*t_alpha*r_cut*r_cut);
 } 
 
 
void FixChargeEwald::setTotalQ2() {
    int nAtoms = state->atoms.size();    
    GPUArrayGlobal<float>tmp(1);
    tmp.memsetByVal(0.0);
    float conversion = state->units.qqr_to_eng;


    accumulate_gpu<float,float, SumSqr, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>
        (
         tmp.getDevData(),
         state->gpd.qs(state->gpd.activeIdx()),
         nAtoms,
         state->devManager.prop.warpSize,
         SumSqr());
    tmp.dataToHost();   
    total_Q2=conversion*tmp.h_data[0]/state->nPerRingPoly;

    tmp.memsetByVal(0.0);

    accumulate_gpu<float,float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(nAtoms/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>
        (
         tmp.getDevData(),
         state->gpd.qs(state->gpd.activeIdx()),
         nAtoms,
         state->devManager.prop.warpSize,
         SumSingle());

    tmp.dataToHost();   
    total_Q=sqrt(conversion)*tmp.h_data[0]/state->nPerRingPoly;   
    
    cout<<"total_Q "<<total_Q<<'\n';
    cout<<"total_Q2 "<<total_Q2<<'\n';
}
double FixChargeEwald::find_optimal_parameters(bool printError){

    int nAtoms = state->atoms.size();    
    L=state->boundsGPU.trace();
    h=make_float3(L.x/sz.x,L.y/sz.y,L.z/sz.z);
//     cout<<"Lx "<<L.x<<'\n';
//     cout<<"hx "<<h.x<<'\n';
//     cout<<"nA "<<nAtoms<<'\n';

//now root solver 
//solving DeltaF_k=DeltaF_real
//        Log(DeltaF_k)=Log(DeltaF_real)
//        Log(DeltaF_k)-Log(DeltaF_real)=0

//lets try secant
    //two initial points
    double x_a=0.0;
    double x_b=4.79853/r_cut;
    
    double y_a=DeltaF_k(x_a)-DeltaF_real(x_a);
    double y_b=DeltaF_k(x_b)-DeltaF_real(x_b);
//           cout<<x_a<<' '<<y_a<<'\n';
//           cout<<x_b<<' '<<y_b<<' '<<DeltaF_real(x_b)<<'\n';

    double tol=1E-5;
    int n_iter=0,max_iter=100;
    while((fabs(y_b)/DeltaF_real(x_b)>tol)&&(n_iter<max_iter)){
      double kinv=(x_b-x_a)/(y_b-y_a);
      y_a=y_b;
      x_a=x_b;
      x_b=x_a-y_a*kinv;
      y_b=DeltaF_k(x_b)-DeltaF_real(x_b);
//       cout<<x_b<<' '<<y_b<<'\n';
      n_iter++;
    }
    if (n_iter==max_iter) cout<<"Ewald RMS Root finder failed, max_iter "<<max_iter<<" reached\n";
    alpha=x_b;
    setEvalWrapper();
    //set orig!
    //alpha = 1.0;
    double error = DeltaF_k(alpha)+DeltaF_real(alpha);
    if (printError) {

        cout<<"Ewald alpha="<<alpha<<'\n';
        cout<<"Ewald RMS error is  "<<error<<'\n';
    }
    return error;
    
    
}

void FixChargeEwald::setParameters(int szx_,int szy_,int szz_,float rcut_,int interpolation_order_)
{
    //for now support for only 2^N sizes
    //TODO generalize for non cubic boxes
    if (rcut_==-1) {
        rcut_ = state->rCut;
    }
    if ((szx_!=32)&&(szx_!=64)&&(szx_!=128)&&(szx_!=256)&&(szx_!=512)&&(szx_!=1024)){
        cout << szx_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
        exit(2);
    }
    if ((szy_!=32)&&(szy_!=64)&&(szy_!=128)&&(szy_!=256)&&(szy_!=512)&&(szy_!=1024)){
        cout << szy_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
        exit(2);
    }
    if ((szz_!=32)&&(szz_!=64)&&(szz_!=128)&&(szz_!=256)&&(szz_!=512)&&(szz_!=1024)){
        cout << szz_ << " is not supported, sorry. Only 2^N grid size works for charge Ewald\n";
        exit(2);
    }
    sz=make_int3(szx_,szy_,szz_);
    r_cut=rcut_;
    cudaMalloc((void**)&FFT_Qs, sizeof(cufftComplex)*sz.x*sz.y*sz.z);

    cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_C2C);

    
    cudaMalloc((void**)&FFT_Ex, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ey, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    cudaMalloc((void**)&FFT_Ez, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
    
    Green_function=GPUArrayGlobal<float>(sz.x*sz.y*sz.z);
    CUT_CHECK_ERROR("setParameters execution failed");
    

    interpolation_order=interpolation_order_;

    malloced = true;

}


void FixChargeEwald::setGridToErrorTolerance(bool printMsg) {
    int3 szOld = sz;
    int nTries = 0;
    double error = find_optimal_parameters(false);
    Vector trace = state->bounds.rectComponents;
    while (nTries < 100 and (error > errorTolerance or error!=error or error < 0)) { //<0 tests for -inf
        Vector sVec = Vector(make_float3(sz));
        Vector ratio = sVec / trace;
        double minRatio = ratio[0];
        int minIdx = 0;
        for (int i=0; i<3; i++) {
            if (ratio[i] < minRatio) {
                minRatio = ratio[i];
                minIdx = i;
            }
        }
        sVec[minIdx] *= 2;
        //sz *= 2;//make_int3(sVec.asFloat3());
        sz = make_int3(sVec.asFloat3());
        error = find_optimal_parameters(false);
        nTries++;
    }
    //DOESN'T REDUCE GRID SIZE EVER
    if (printMsg) {
        printf("Using ewald grid of %d %d %d with error %f\n", sz.x, sz.y, sz.z, error);
    }

    if (!malloced or szOld != sz) {
        if (malloced) {
            cufftDestroy(plan);
            cudaFree(FFT_Qs);
            cudaFree(FFT_Ex);
            cudaFree(FFT_Ey);
            cudaFree(FFT_Ez);
        }
        cudaMalloc((void**)&FFT_Qs, sizeof(cufftComplex)*sz.x*sz.y*sz.z);

        cufftPlan3d(&plan, sz.x,sz.y, sz.z, CUFFT_C2C);


        cudaMalloc((void**)&FFT_Ex, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
        cudaMalloc((void**)&FFT_Ey, sizeof(cufftComplex)*sz.x*sz.y*sz.z);
        cudaMalloc((void**)&FFT_Ez, sizeof(cufftComplex)*sz.x*sz.y*sz.z);

        Green_function=GPUArrayGlobal<float>(sz.x*sz.y*sz.z);
        malloced = true;
    }


}
void FixChargeEwald::setError(double targetError, float rcut_, int interpolation_order_) {
    if (rcut_==-1) {
        rcut_ = state->rCut;
    }
    r_cut=rcut_;
    interpolation_order=interpolation_order_;
    errorTolerance = targetError;
    modeIsError = true;

}

void FixChargeEwald::calc_Green_function(){

    
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    int sum_limits=int(alpha*pow(h.x*h.y*h.z,1.0/3.0)/3.14159*(sqrt(-log(10E-7))))+1;
    Green_function_cu<<<dimGrid, dimBlock>>>(state->boundsGPU, sz,Green_function.getDevData(),alpha,
                                             sum_limits,interpolation_order);//TODO parameters unknown
    CUT_CHECK_ERROR("Green_function_cu kernel execution failed");
    
        //test area
//     Green_function.dataToHost();
//     ofstream ofs;
//     ofs.open("test_Green_function.dat",ios::out );
//     for(int i=0;i<sz.x;i++)
//             for(int j=0;j<sz.y;j++){
//                 for(int k=0;k<sz.z;k++){
//                     cout<<Green_function.h_data[i*sz.y*sz.z+j*sz.z+k]<<'\t';
//                     ofs<<Green_function.h_data[i*sz.y*sz.z+j*sz.z+k]<<'\t';
//                 }
//                 ofs<<'\n';
//                 cout<<'\n';
//             }
//     ofs.close();

}


void FixChargeEwald::calc_potential(cufftComplex *phi_buf){
     BoundsGPU b=state->boundsGPU;
    float volume=b.volume();
    
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    potential_cu<<<dimGrid, dimBlock>>>(sz,Green_function.getDevData(), FFT_Qs,phi_buf);
    CUT_CHECK_ERROR("potential_cu kernel execution failed");    


    cufftExecC2C(plan, phi_buf, phi_buf,  CUFFT_INVERSE);
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C execution failed");

//     //test area
//     float *buf=new float[sz.x*sz.y*sz.z*2];
//     cudaMemcpy((void *)buf,phi_buf,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
//     ofstream ofs;
//     ofs.open("test_phi.dat",ios::out );
//     for(int i=0;i<sz.x;i++)
//             for(int j=0;j<sz.y;j++){
//                 for(int k=0;k<sz.z;k++){
//                     cout<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
//                      ofs<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
//                 }
//                 ofs<<'\n';
//                 cout<<'\n';
//             }
//     ofs.close();
//     delete []buf;
}

bool FixChargeEwald::prepareForRun() {
    virialField = GPUArrayDeviceGlobal<Virial>(1);
    setTotalQ2();

    //TODO these values for comparison are uninitialized - we should see about this

    handleBoundsChangeInternal(true);
    turnInit = state->turn;
    if (longRangeInterval != 1) {
        storedForces = GPUArrayDeviceGlobal<float4>(state->maxIdExisting+1);
    } else {
        storedForces = GPUArrayDeviceGlobal<float4>(1);
    }
    if (state->nPerRingPoly > 1) { 
        rpCentroids = GPUArrayDeviceGlobal<float4>(state->atoms.size() / state->nPerRingPoly);
    }
    setEvalWrapper();
    prepared = true;
    return prepared;
}

void FixChargeEwald::setEvalWrapper() {
    if (evalWrapperMode == "offload") {
        if (hasOffloadedChargePairCalc) {
            evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), nullptr); //nParams arg is 1 rather than zero b/c can't have zero sized argument on device
        } else {
            evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), this);
        }
    } else if (evalWrapperMode == "self") {
        evalWrap = pickEvaluator<EvaluatorNone, 1, false>(EvaluatorNone(), this);
    }

}


void FixChargeEwald::handleBoundsChange() {
    handleBoundsChangeInternal(false);
}

void FixChargeEwald::handleBoundsChangeInternal(bool printError) {

    if ((state->boundsGPU != boundsLastOptimize)||(total_Q2!=total_Q2LastOptimize)) {
        if (modeIsError) {
            setGridToErrorTolerance(printError);
        } else {
            find_optimal_parameters(printError);
        }
        calc_Green_function();
        boundsLastOptimize = state->boundsGPU;
        total_Q2LastOptimize=total_Q2;
    }
}

void FixChargeEwald::compute(int virialMode) {
 //   CUT_CHECK_ERROR("before FixChargeEwald kernel execution failed");

//     cout<<"FixChargeEwald::compute..\n";
    int nAtoms       = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int nRingPoly    = nAtoms / nPerRingPoly;
    GPUData &gpd     = state->gpd;
    GridGPU &grid    = state->gridGPU;
    int activeIdx    = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
 
    float Qconversion = sqrt(state->units.qqr_to_eng);

    //first update grid from atoms positions
    //set qs to 0
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    if (not ((state->turn - turnInit) % longRangeInterval)) {
        map_charge_set_to_zero_cu<<<dimGrid, dimBlock>>>(sz,FFT_Qs);
        //  CUT_CHECK_ERROR("map_charge_set_to_zero_cu kernel execution failed");

        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        // Compute centroids of all ring polymers for use on grid
        //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        float4 *centroids;
        BoundsGPU bounds         = state->boundsGPU;
        BoundsGPU boundsUnskewed = bounds.unskewed();
        if (nPerRingPoly >1) {
            computeCentroid<<<NBLOCK(nRingPoly),PERBLOCK>>>(rpCentroids.data(),gpd.xs(activeIdx),nAtoms,nPerRingPoly,boundsUnskewed);
            centroids = rpCentroids.data();
        } else {
            centroids = gpd.xs(activeIdx);
        }
        switch (interpolation_order){
            case 1:{map_charge_to_grid_order_1_cu
                       <<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly, 
                               centroids,                                                      
                               gpd.qs(activeIdx),
                               state->boundsGPU,
                               sz,
                               (float *)FFT_Qs,
                               Qconversion);
                       break;}
            case 3:{map_charge_to_grid_order_3_cu
                       <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly,
                               centroids,
                               gpd.qs(activeIdx),
                               state->boundsGPU,
                               sz,
                               (float *)FFT_Qs,
                               Qconversion);
                       break;}
        }    
        // CUT_CHECK_ERROR("map_charge_to_grid_cu kernel execution failed");

        cufftExecC2C(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);
        // cudaDeviceSynchronize();
        //  CUT_CHECK_ERROR("cufftExecC2C Qs execution failed");


        //     //test area
        //     float buf[sz.x*sz.y*sz.z*2];
        //     cudaMemcpy(buf,FFT_Qs,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
        //     ofstream ofs;
        //     ofs.open("test_FFT.dat",ios::out );
        //     for(int i=0;i<sz.x;i++)
        //             for(int j=0;j<sz.y;j++){
        //                 for(int k=0;k<sz.z;k++){
        //                     cout<<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]<<'\t';
        //                     ofs <<buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]<<'\t';
        //                 }
        //                 ofs<<'\n';
        //                 cout<<'\n';
        //             }


        //next potential calculation: just going to use Ex to store it for now
        //       calc_potential(FFT_Ex);

        //calc E field
        E_field_cu<<<dimGrid, dimBlock>>>(state->boundsGPU,sz,Green_function.getDevData(), FFT_Qs,FFT_Ex,FFT_Ey,FFT_Ez);
        CUT_CHECK_ERROR("E_field_cu kernel execution failed");    


        cufftExecC2C(plan, FFT_Ex, FFT_Ex,  CUFFT_INVERSE);
        cufftExecC2C(plan, FFT_Ey, FFT_Ey,  CUFFT_INVERSE);
        cufftExecC2C(plan, FFT_Ez, FFT_Ez,  CUFFT_INVERSE);
        //  cudaDeviceSynchronize();
        // CUT_CHECK_ERROR("cufftExecC2C  E_field execution failed");


        /*//test area
          Bounds b=state->bounds;
          float volume=b.trace[0]*b.trace[1]*b.trace[2];    
          float *buf=new float[sz.x*sz.y*sz.z*2];
          cudaMemcpy((void *)buf,FFT_Ex,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
          ofstream ofs;
          ofs.open("test_Ex.dat",ios::out );
          for(int i=0;i<sz.x;i++)
          for(int j=0;j<sz.y;j++){
          for(int k=0;k<sz.z;k++){
          cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          }
          ofs<<'\n';
          cout<<'\n';
          }
          ofs.close();
          cudaMemcpy((void *)buf,FFT_Ey,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
          ofs.open("test_Ey.dat",ios::out );
          for(int i=0;i<sz.x;i++)
          for(int j=0;j<sz.y;j++){
          for(int k=0;k<sz.z;k++){
          cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          }
          ofs<<'\n';
          cout<<'\n';
          }
          ofs.close();    
          cudaMemcpy((void *)buf,FFT_Ez,sizeof(cufftComplex)*sz.x*sz.y*sz.z,cudaMemcpyDeviceToHost );
          ofs.open("test_Ez.dat",ios::out );
          for(int i=0;i<sz.x;i++)
          for(int j=0;j<sz.y;j++){
          for(int k=0;k<sz.z;k++){
          cout<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          ofs<<-buf[i*sz.y*sz.z*2+j*sz.z*2+k*2]/volume<<'\t';
          }
          ofs<<'\n';
          cout<<'\n';
          }
          ofs.close();    
          delete []buf;   */ 




        //calc forces
        //printf("Forces!\n");
        // Performing an "effective" ring polymer contraction means that we should evaluate the forces
        // for the centroids
        bool storeForces = longRangeInterval != 1;
        switch (interpolation_order){
            case 1:{Ewald_long_range_forces_order_1_cu<<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly,
                           centroids,                                                      
                           gpd.fs(activeIdx),
                           gpd.qs(activeIdx),
                           state->boundsGPU,
                           sz,
                           FFT_Ex,FFT_Ey,FFT_Ez,Qconversion,
                           storeForces, gpd.ids(activeIdx), storedForces.data()
                           );
                       break;}
            case 3:{Ewald_long_range_forces_order_3_cu<<<NBLOCK(nRingPoly), PERBLOCK>>>( nRingPoly, nPerRingPoly,
                           centroids,                                                      
                           gpd.fs(activeIdx),
                           gpd.qs(activeIdx),
                           state->boundsGPU,
                           sz,
                           FFT_Ex,FFT_Ey,FFT_Ez,Qconversion,
                           storeForces, gpd.ids(activeIdx), storedForces.data()
                           );

                       break;}
        }
    } else {
        applyStoredForces<<<NBLOCK(nAtoms), PERBLOCK>>>( nAtoms,
                gpd.fs(activeIdx),
                gpd.ids(activeIdx), storedForces.data());
    }
    CUT_CHECK_ERROR("Ewald_long_range_forces_cu  execution failed");
    //SHORT RANGE
    if (virialMode) {
        int warpSize = state->devManager.prop.warpSize;
        BoundsGPU &b=state->boundsGPU;
        float volume=b.volume();          
        virialField.memset(0); 
        virials_cu<<<dimGrid, dimBlock,sizeof(Virial)*dimBlock.x*dimBlock.y*dimBlock.z>>>(state->boundsGPU,sz,virialField.data(),alpha,Green_function.getDevData(), FFT_Qs, warpSize); 
        CUT_CHECK_ERROR("virials_cu kernel execution failed");    



        mapVirialToSingleAtom<<<1, 6>>>(gpd.virials.d_data.data(), virialField.data(), volume);
    }

    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->compute(nAtoms,nPerRingPoly,gpd.xs(activeIdx), gpd.fs(activeIdx),
                  neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(),
                  state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU, //PASSING NULLPTR TO GPU MAY CAUSE ISSUES
    //ALTERNATIVELy, COULD JUST GIVE THE PARMS SOME OTHER RANDOM POINTER, AS LONG AS IT'S VALID
                  neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.virials.d_data.data(), gpd.qs(activeIdx), r_cut, virialMode, nThreadPerBlock(), nThreadPerAtom());


    CUT_CHECK_ERROR("Ewald_short_range_forces_cu  execution failed");

}


void FixChargeEwald::singlePointEng(float * perParticleEng) {
    CUT_CHECK_ERROR("before FixChargeEwald kernel execution failed");

    if (state->boundsGPU != boundsLastOptimize) {
        handleBoundsChange();
    }
//     cout<<"FixChargeEwald::compute..\n";
    int nAtoms = state->atoms.size();
    int nPerRingPoly = state->nPerRingPoly;
    int nRingPoly    = nAtoms / nPerRingPoly;
    GPUData &gpd = state->gpd;
    GridGPU &grid = state->gridGPU;
    int activeIdx = gpd.activeIdx();
    uint16_t *neighborCounts = grid.perAtomArray.d_data.data();
    
    
     
    float Qconversion = sqrt(state->units.qqr_to_eng);


    //first update grid from atoms positions
    //set qs to 0
    float field_energy_per_particle = 0;
    dim3 dimBlock(8,8,8);
    dim3 dimGrid((sz.x + dimBlock.x - 1) / dimBlock.x,(sz.y + dimBlock.y - 1) / dimBlock.y,(sz.z + dimBlock.z - 1) / dimBlock.z);    
    map_charge_set_to_zero_cu<<<dimGrid, dimBlock>>>(sz,FFT_Qs);
    CUT_CHECK_ERROR("map_charge_set_to_zero_cu kernel execution failed");
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Compute centroids of all ring polymers for use on grid
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    float4 *centroids;
    BoundsGPU bounds         = state->boundsGPU;
    BoundsGPU boundsUnskewed = bounds.unskewed();
    if (nPerRingPoly >1) {
        rpCentroids = GPUArrayDeviceGlobal<float4>(nRingPoly);
        computeCentroid<<<NBLOCK(nRingPoly),PERBLOCK>>>(rpCentroids.data(),gpd.xs(activeIdx),nAtoms,nPerRingPoly,boundsUnskewed);
        centroids = rpCentroids.data();
        periodicWrapCpy<<<NBLOCK(nRingPoly), PERBLOCK>>>(centroids, nRingPoly, boundsUnskewed);
    } else {
        centroids = gpd.xs(activeIdx);
    }

      switch (interpolation_order){
      case 1:{map_charge_to_grid_order_1_cu
              <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly,
                                              centroids,
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (float *)FFT_Qs,Qconversion);
              break;}
      case 3:{map_charge_to_grid_order_3_cu
              <<<NBLOCK(nRingPoly), PERBLOCK>>>(nRingPoly, nPerRingPoly, 
                                              centroids,
                                              gpd.qs(activeIdx),
                                              state->boundsGPU,
                                              sz,
                                              (float *)FFT_Qs,Qconversion);
              break;}
    }    
    CUT_CHECK_ERROR("map_charge_to_grid_cu kernel execution failed");

    cufftExecC2C(plan, FFT_Qs, FFT_Qs, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    CUT_CHECK_ERROR("cufftExecC2C Qs execution failed");

    

    //calc field energy 
    BoundsGPU &b=state->boundsGPU;
    float volume=b.volume();
    
    Energy_cu<<<dimGrid, dimBlock>>>(sz,Green_function.getDevData(), FFT_Qs,FFT_Ex);//use Ex as buffer
    CUT_CHECK_ERROR("Energy_cu kernel execution failed");    
  
    GPUArrayGlobal<float>field_E(1);
    field_E.memsetByVal(0.0);
    int warpSize = state->devManager.prop.warpSize;
    accumulate_gpu<float,float, SumSingle, N_DATA_PER_THREAD> <<<NBLOCK(2*sz.x*sz.y*sz.z/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>
        (
         field_E.getDevData(),
         (float *)FFT_Ex,
         2*sz.x*sz.y*sz.z,
         warpSize,
         SumSingle()
         );   
/*
    sumSingle<float,float, N_DATA_PER_THREAD> <<<NBLOCK(2*sz.x*sz.y*sz.z/(double)N_DATA_PER_THREAD),PERBLOCK,N_DATA_PER_THREAD*sizeof(float)*PERBLOCK>>>(
                                            field_E.getDevData(),
                                            (float *)FFT_Ex,
                                            2*sz.x*sz.y*sz.z,
                                            warpSize);   
                                            */
    field_E.dataToHost();

    //field_energy_per_particle=0.5*field_E.h_data[0]/volume/nAtoms;
    field_energy_per_particle=0.5*field_E.h_data[0]/volume/nRingPoly;
//         cout<<"field_E "<<field_E.h_data[0]<<'\n';

    field_energy_per_particle-=alpha/sqrt(M_PI)*total_Q2/nRingPoly;
//      cout<<"self correction "<<alpha/sqrt(M_PI)*total_Q2<<'\n';

//pair energies
    mapEngToParticles<<<NBLOCK(nAtoms), PERBLOCK>>>(nAtoms, field_energy_per_particle, perParticleEng);
    float *neighborCoefs = state->specialNeighborCoefs;
    evalWrap->energy(nAtoms,nPerRingPoly, gpd.xs(activeIdx), perParticleEng, neighborCounts, grid.neighborlist.data(), grid.perBlockArray.d_data.data(), state->devManager.prop.warpSize, nullptr, 0, state->boundsGPU, neighborCoefs[0], neighborCoefs[1], neighborCoefs[2], gpd.qs(activeIdx), r_cut, nThreadPerBlock(), nThreadPerAtom());


    CUT_CHECK_ERROR("Ewald_short_range_forces_cu  execution failed");

}


int FixChargeEwald::setLongRangeInterval(int interval) {
    if (interval) {
        longRangeInterval = interval;
    }
    return longRangeInterval;
}



ChargeEvaluatorEwald FixChargeEwald::generateEvaluator() {
    return ChargeEvaluatorEwald(alpha, state->units.qqr_to_eng);
}

void (FixChargeEwald::*setParameters_xyz)(int ,int ,int ,float ,int) = &FixChargeEwald::setParameters;
void (FixChargeEwald::*setParameters_xxx)(int ,float ,int) = &FixChargeEwald::setParameters;
void export_FixChargeEwald() {
    py::class_<FixChargeEwald,
                          SHARED(FixChargeEwald),
                          py::bases<FixCharge> > (
         "FixChargeEwald", 
         py::init<SHARED(State), string, string> (
              py::args("state", "handle", "groupHandle"))
        )
        .def("setParameters", setParameters_xyz,
                (py::arg("szx"),py::arg("szy"),py::arg("szz"), py::arg("r_cut")=-1,py::arg("interpolation_order")=3)
          
            )
        .def("setParameters", setParameters_xxx,
                (py::arg("sz"),py::arg("r_cut")=-1,py::arg("interpolation_order")=3)
            )        
        .def("setError", &FixChargeEwald::setError, (py::arg("error"), py::arg("rCut")=-1, py::arg("interpolation_order")=3)
            )
        .def("setLongRangeInterval", &FixChargeEwald::setLongRangeInterval, (py::arg("interval")=0))
        ;
}

