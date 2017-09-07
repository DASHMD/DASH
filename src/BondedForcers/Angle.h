#pragma once
#ifndef ANGLE_H
#define ANGLE_H
#include "globalDefs.h"
#include "Atom.h"

#include "cutils_math.h"
#include <boost/variant.hpp>
#include <array>
class AngleHarmonic;
class AngleCHARMM;
class AngleCosineDelta;
void export_AngleHarmonic();
void export_AngleCHARMM();
void export_AngleCosineDelta();

class Angle {
    public:
        //going to try storing by id instead.  Makes preparing for a run less intensive
        int type;
        std::array<int, 3> ids;
        void takeIds(Angle *);
	std::string getInfoString();
};

class AngleGPU {
    public:
        int ids[3];
        uint32_t type; //myIdx (which atom in these three we're actually calcing the for for) is stored in two left-most bits
        void takeIds(Angle *);


};
//angle harmonic
class AngleHarmonicType {
public:
    float k;
    float theta0;
    AngleHarmonicType(AngleHarmonic *);
    AngleHarmonicType(){};
    bool operator==(const AngleHarmonicType &) const;
	std::string getInfoString();
};

class AngleHarmonic : public Angle, public AngleHarmonicType {
public:
    AngleHarmonic(Atom *a, Atom *b, Atom *c, double k_, double theta0_, int type_=-1);
    AngleHarmonic(double k_, double theta0_, int type_=-1);
    AngleHarmonic(){};
    int type;
	std::string getInfoString();
};

//for forcer maps
namespace std {
    template<> struct hash<AngleHarmonicType> {
        size_t operator() (AngleHarmonicType const& ang) const {
            size_t seed = 0;
            boost::hash_combine(seed, ang.k);
            boost::hash_combine(seed, ang.theta0);
            return seed;
        }
    };
}

//angle CHARMM
class AngleCHARMMType {
public:
    float k;
    float theta0;
    float kub;
    float rub;
    AngleCHARMMType(AngleCHARMM *);
    AngleCHARMMType(){};
    bool operator==(const AngleCHARMMType &) const;
	std::string getInfoString();
};

class AngleCHARMM : public Angle, public AngleCHARMMType {
public:
    AngleCHARMM(Atom *a, Atom *b, Atom *c, double k_  , double theta0_, double kub_, double rub_,int type_=-1);
    AngleCHARMM(double k_, double theta0_, double kub_, double rub_, int type_=-1);
    AngleCHARMM(){};
    int type;
	std::string getInfoString();
};

//for forcer maps
namespace std {
    template<> struct hash<AngleCHARMMType> {
        size_t operator() (AngleCHARMMType const& ang) const {
            size_t seed = 0;
            boost::hash_combine(seed, ang.k);
            boost::hash_combine(seed, ang.theta0);
            boost::hash_combine(seed, ang.kub);
            boost::hash_combine(seed, ang.rub);
            return seed;
        }
    };
}

//angle cosine delta
class AngleCosineDeltaType {
public:
    float k;
    float theta0;
    AngleCosineDeltaType(AngleCosineDelta *);
    AngleCosineDeltaType(){};
    bool operator==(const AngleCosineDeltaType &) const;
	std::string getInfoString();
};

class AngleCosineDelta : public Angle, public AngleCosineDeltaType {
public:
    AngleCosineDelta(Atom *a, Atom *b, Atom *c, double k_, double theta0_, int type_=-1);
    AngleCosineDelta(double k_, double theta0_, int type_=-1); 
    AngleCosineDelta(){};
    int type;
	std::string getInfoString();
};

//for forcer maps
namespace std {
    template<> struct hash<AngleCosineDeltaType> {
        size_t operator() (AngleCosineDeltaType const& ang) const {
            size_t seed = 0;
            boost::hash_combine(seed, ang.k);
            boost::hash_combine(seed, ang.theta0);
            return seed;
        }
    };
}



// lets us store a list of vectors to any kind of angles we want
typedef boost::variant<
	AngleHarmonic, 
	AngleCHARMM, 
    AngleCosineDelta,
    Angle	
> AngleVariant;
#endif
