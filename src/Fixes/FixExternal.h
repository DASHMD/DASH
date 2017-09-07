#pragma once

#ifndef FIX_EXTERNAL_H
#define FIX_EXTERNAL_H

#include <map>
#include <string>
#include <vector>

#include "GPUArrayGlobal.h"
#include "Fix.h"

class State;

void export_FixExternal();

class FixExternal: public Fix {

public:
        // General variables for the class

	// constructor
	FixExternal(SHARED(State) state_, std::string handle_, 
			std::string groupHandle_, std::string type_, 
			bool forceSingle_, bool requiresCharges_, int applyEvery_)
		: Fix(state_, handle_, groupHandle_, type_, true, false, false, applyEvery_)
		{
		
       		 };
};

#endif
