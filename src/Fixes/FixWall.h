#pragma once

#ifndef FIX_WALL_H
#define FIX_WALL_H

#include <map>
#include <string>
#include <vector>

#include "GPUArrayGlobal.h"
#include "Fix.h"

class State;

void export_FixWall();

class FixWall: public Fix {

public:

	// constructor
	FixWall(SHARED(State) state_, std::string handle_, 
			std::string groupHandle_, std::string type_, 
			bool forceSingle_, bool requiresCharges_, int applyEvery_,
            Vector origin_, Vector forceDir_)
		: Fix(state_, handle_, groupHandle_, type_, true, false, false, 
		applyEvery_), origin(origin_), forceDir(forceDir_)

		{
		
        };

	// all will have origin
	Vector origin;
	// direction the force projects	
	Vector forceDir;
    
    // the dist will be put in the specific fix, not the base FixWall class
	

};

#endif
