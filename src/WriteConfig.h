#pragma once
#ifndef WRITECONFIG_H
#define WRITECONFIG_H

#include <fstream>
#include <sstream>

#include "State.h"
#include "base64.h"
#include "boost_for_export.h"

#define FN_LEN 150

void export_WriteConfig();

class WriteConfig {

private:
    void (*writeFormat)(State *, std::string, int64_t, bool, uint);    

public:
    State *state;
    std::string fn;
    std::string handle;
    std::string format;
	std::string groupHandle;
    int writeEvery;
	int groupBit;
    
    bool andVelocities;

    bool isXML;
    bool unwrapMolecules;
    int orderPreference; //just there so I can use same functions as fix for adding/removing
    bool oneFilePerWrite;


    WriteConfig(boost::shared_ptr<State>,
                std::string fn, std::string handle, std::string format, int writeEvery, std::string groupHandle_="all", bool unwrapMolecules=false);

    ~WriteConfig() {
        finish();
    }
    void unwrap();
    void finish();

    void write(int64_t turn);
    void writePy();
    std::string getCurrentFn(int64_t turn);

};

#endif
