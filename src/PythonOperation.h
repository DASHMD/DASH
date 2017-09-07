#pragma once
#ifndef PYTHONOPERATION_H
#define PYTHONOPERATION_H

#include "globalDefs.h"

#include "boost_for_export.h"
#include <string>
void export_PythonOperation();

class PythonOperation {
    public:
        bool operate(int64_t turn);
        int orderPreference; // needed for add generic in state, not actually used
        PyObject *operation;
        int operateEvery; 
        std::string handle;
        bool synchronous;
        PythonOperation(std::string, int, PyObject*, bool synchronous_=false);
    //OKAY, so I would like to have it so that you can set the next turn when this is called arbitrarily, but then
    //if you have pyOp return next turn so like user decides when next turn is based on current operation,
    //then it's dangerous, b/c you may have already passed that turn!
    //Better to have like a seperate pyfunc which you can call repeatedly and it gives the next turn, then you keep 
    //track of next turn, following ones, etc. Can implement this later if need be.  

};

#endif
