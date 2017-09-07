#pragma once
#ifndef FIXWALLLJ126_H
#define FIXWALLLJ126_H

#include "FixWall.h"
#include "WallEvaluatorLJ126.h"

void export_FixWallLJ126();

class FixWallLJ126 : public FixWall {

    public: 
        
		FixWallLJ126(SHARED(State), std::string handle_, std::string groupHandle_,
							Vector origin_, Vector forceDir_, float dist_, float sigma_, float epsilon_);
		float dist;

		float sigma;
		
        float epsilon;

        void compute(int);

		bool prepareForRun();
		
        bool postRun();

        void singlePointEng(float *);
		
        EvaluatorWallLJ126 evaluator; // evaluator for LJ 12-6 wall interactions
};

#endif
