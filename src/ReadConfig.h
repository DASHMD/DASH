#pragma once
#ifndef READCONFIG_H
#define READCONFIG_H

#include <string>
#include <sstream>

#include "Python.h"
#include <boost/shared_ptr.hpp>
#include <pugixml.hpp>

#include "globalDefs.h"
#include "boost_for_export.h"

void export_ReadConfig();

class State;

class ReadConfig {

private:
    std::string fn;
    State *state;
    bool haveReadYet;
    boost::shared_ptr<pugi::xml_document> doc;  // doing pointers b/c copy semantics for these are weird
    boost::shared_ptr<pugi::xml_node> config;

    bool read();

public:
    bool fileOpen;

    ReadConfig()
      : state(nullptr)
    {   }
    ReadConfig(State *state_);

    pugi::xml_node readNode(std::string nodeTag);

    void loadFile(std::string);  // change to bool or something to give feedback about if it's a file or not
    bool next();
    bool prev();
    bool moveBy(int);
    pugi::xml_node readFix(std::string type, std::string handle);
    //bool readConfig(boost::shared_ptr<State>, std::string, int configIdx=0);

};

#endif
