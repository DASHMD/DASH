#pragma once
#ifndef IMPROPER_H
#define IMPROPER_H

#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <boost/functional/hash.hpp>
#include <array>
void export_Impropers();
class Improper {
public:

    std::array<int, 4> ids;
    int type;
    void takeIds(Improper *);
    std::string getInfoString();
};



class ImproperHarmonic : public Improper {
public:
    double k;
    double thetaEq;
    ImproperHarmonic(Atom *a, Atom *b, Atom *c, Atom *d, double k, double thetaEq, int type_=-1);
    ImproperHarmonic(double k, double thetaEq, int type_=-1);
    ImproperHarmonic(){};
    std::string getInfoString();
};

class ImproperHarmonicType {
public:
    float thetaEq;
    float k;
    ImproperHarmonicType(ImproperHarmonic *);
    ImproperHarmonicType(){}; //for hashing, need default constructor, == operator, and std::hash function
    bool operator==(const ImproperHarmonicType &) const;
    std::string getInfoString();
};



class ImproperCVFF: public Improper {
public:
    double k;
    int d;
    int n;
    ImproperCVFF(Atom *a, Atom *b, Atom *c, Atom *d, double k, int dParam, int n, int type_=-1);
    ImproperCVFF(double k, int d, int n, int type_=-1);
    ImproperCVFF(){};
    std::string getInfoString();
};

class ImproperCVFFType {
public:
    float k;
    int d;
    int n;
    ImproperCVFFType(ImproperCVFF *);
    ImproperCVFFType(){}; //for hashing, need default constructor, == operator, and std::hash function
    bool operator==(const ImproperCVFFType &) const;
    std::string getInfoString();
};




class ImproperGPU{
public:
    int ids[4];
    uint32_t type;
    void takeIds(Improper *);


};
//for forcer maps
namespace std {
template<> struct hash<ImproperHarmonicType> {
    size_t operator() (ImproperHarmonicType const& imp) const {
        size_t seed = 0;
        boost::hash_combine(seed, imp.k);
        boost::hash_combine(seed, imp.thetaEq);
        return seed;
    }
};

template<> struct hash<ImproperCVFFType> {
    size_t operator() (ImproperCVFFType const& imp) const {
        size_t seed = 0;
        boost::hash_combine(seed, imp.k);
        boost::hash_combine(seed, imp.d);
        boost::hash_combine(seed, imp.n);
        return seed;
    }
};


}
typedef boost::variant<
ImproperHarmonic, 
ImproperCVFF, 
    Improper	
    > ImproperVariant;

#endif
