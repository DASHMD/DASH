#pragma once
#ifndef FIXRIGID_H
#define FIXRIGID_H

#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include "Python.h"
#include "Fix.h"
#include "FixBond.h"
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python/list.hpp>
#include "GPUArrayDeviceGlobal.h"

//void settle_xs(float timestep, float3 com, float3 com1, float3 *xs_0, float3 *xs, float3 *fix_len);
//void settle_vs(float timestep, float3 *vs_0, float3 *vs, float3 *xs, float *mass, float3 *fix_len);

void export_FixRigid();

class FixRigid : public Fix {
 private:
  GPUArrayDeviceGlobal<int4> waterIdsGPU;
  GPUArrayDeviceGlobal<float4> xs_0;
  GPUArrayDeviceGlobal<float4> vs_0;
  GPUArrayDeviceGlobal<float4> dvs_0;
  GPUArrayDeviceGlobal<float4> fs_0;
  GPUArrayDeviceGlobal<float4> com;
  GPUArrayDeviceGlobal<float4> fix_len;
  std::vector<int4> waterIds;
  std::vector<BondVariant> bonds;
  std::vector<float4> invMassSums;
  bool fourSet = false;
  bool fourSite;

 public:
  FixRigid(SHARED(State), std::string handle_, std::string groupHandle_);
  bool stepInit();
  bool stepFinal();
  bool prepareForRun();
  void createRigid(int, int, int);
  //void createRigidTIP4P(int, int, int, int);

  std::vector<BondVariant> *getBonds() {
    return &bonds;
  }
};

#endif
