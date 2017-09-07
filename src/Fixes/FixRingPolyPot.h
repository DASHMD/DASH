#pragma once
#ifndef FIXRINGPOLYPOT_H
#define FIXRINGPOLYPOT_H

#include "Fix.h"
#include "GPUArrayGlobal.h"
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
#include <string>
#include <vector>


void export_FixRingPolyPot();

class FixRingPolyPot : public Fix {
	public:
		FixRingPolyPot(SHARED(State), std::string handle_, std::string groupHandle_);
		
        void singlePointEng(float *);
};

#endif
