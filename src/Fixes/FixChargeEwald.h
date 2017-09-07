#pragma once
#include <cufft.h>

//#include "AtomParams.h"
#include "FixCharge.h"
#include "GPUArrayGlobal.h"
#include "Virial.h"
#include "BoundsGPU.h"
#include "ChargeEvaluatorEwald.h"

class State;

void export_FixChargeEwald();
extern const std::string chargeEwaldType;

/*! \class FixChargeEwald
 * \brief Short and Long range Coulomb interaction
 * Short range interactions are computed pairwise manner
 * Long range interactions are computed via Fourier space
 * Implementation based on Deserno and Holm, J.Chem.Phys. 109, 7678
 *
 */
class FixChargeEwald : public FixCharge {

private:
    cufftHandle plan;
    cufftComplex *FFT_Qs;  // change to GPU arrays?
    cufftComplex *FFT_Ex, *FFT_Ey, *FFT_Ez;
    
    GPUArrayGlobal<float> Green_function;  // Green function in k space


    int3 sz;

    float alpha;
    float r_cut;
    
    double find_optimal_parameters(bool);
    
    float total_Q;
    float total_Q2;
    void setTotalQ2();
    void calc_Green_function();
    void calc_potential(cufftComplex *phi_buf);

    int interpolation_order;
//! RMS variables
    double DeltaF_k(double t_alpha);
    double DeltaF_real(double t_alpha);
    float3 h;
    float3 L;
    GPUArrayDeviceGlobal<Virial> virialField;
    GPUArrayDeviceGlobal<float4> storedForces;
    BoundsGPU boundsLastOptimize;
    float total_Q2LastOptimize;    
    void handleBoundsChangeInternal(bool);
    void setGridToErrorTolerance(bool);
    bool modeIsError;
    double errorTolerance;
        
    bool malloced;


public:
    GPUArrayDeviceGlobal<float4> rpCentroids;
    int longRangeInterval;
    int64_t turnInit;
    void handleBoundsChange();
    FixChargeEwald(boost::shared_ptr<State> state_,
                   std::string handle_, std::string groupHandle_);
    ~FixChargeEwald();

    void setError(double error, float rcut_, int interpolation_order_);
    void setParameters(int szx_, int szy_, int szz_, float rcut_, int interpolation_order_);
    void setParameters(int sz_, float rcut_, int interpolation_order_) {
        setParameters(sz_, sz_, sz_, rcut_, interpolation_order_);
    }

    //! Compute forces
    void compute(int);
    int setLongRangeInterval(int interval);

    //! Compute single point energy
    void singlePointEng(float *);
    //void singlePointEngGroupGroup(float *, uint32_t, uint32_t);

    bool prepareForRun();
    
    //! Return list of cutoff values.
    std::vector<float> getRCuts() {
        std::vector<float> res;
        res.push_back(r_cut);
        return res;
    }    

    ChargeEvaluatorEwald generateEvaluator();
    void setEvalWrapper();
};

