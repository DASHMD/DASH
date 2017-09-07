#pragma once
#ifndef BOND_H
#define BOND_H

#include <boost/variant.hpp>

#include "globalDefs.h"
#include "Atom.h"
#include <array>



/*! \brief Bond connecting atoms
 *
 * \link Atom Atoms\endlink can be connected by Bonds. Bonds are defined by a
 * potential depending on the separation of the two bonded \link Atom
 * Atoms\endlink. 
 */
class BondHarmonic;

class Bond {
    public:
        std::array<int, 2> ids;//!<atom ids
        int type; //!< Bond type
};

//bond harmonic classes
//
class BondHarmonicType {
public:
    float k;
    float r0;
 //   BondHarmonicType(BondHarmonic *);
    BondHarmonicType(){};
    bool operator==(const BondHarmonicType &) const;
    std::string getInfoString();
};
//
//for forcer maps
namespace std {
    template<> struct hash<BondHarmonicType> {
        size_t operator() (BondHarmonicType const& bond) const {
            size_t seed = 0;
            boost::hash_combine(seed, bond.k);
            boost::hash_combine(seed, bond.r0);
            return seed;
        }
    };
}

/*! \brief Bond with a harmonic potential (a spring)
 *
 * Bond with harmonic potential.
 *
 * \todo In LAMMPS k is, in fact, k/2. Specify this explicitely here.
 */





class BondHarmonic : public Bond, public BondHarmonicType {
	public:
        BondHarmonic(Atom *a, Atom *b, double k_, double r0_, int type_=-1);
        BondHarmonic(double k_, double r0_, int type_=-1); //is this constructor used?
        BondHarmonic(){};
        int type;
	std::string getInfoString();
};	

void export_BondHarmonic();
//end bond harmonic classes

//bond quartic classes
//
class BondQuarticType {
public:
    float k2;
    float k3;
    float k4;
    float r0;
    BondQuarticType(){};
    bool operator==(const BondQuarticType &) const;
    std::string getInfoString();
};
//
//for forcer maps
namespace std {
    template<> struct hash<BondQuarticType> {
        size_t operator() (BondQuarticType const& bond) const {
            size_t seed = 0;
            boost::hash_combine(seed, bond.k2);
            boost::hash_combine(seed, bond.k3);
            boost::hash_combine(seed, bond.k4);
            boost::hash_combine(seed, bond.r0);
            return seed;
        }
    };
}

/*! \brief Bond with a Quartic potential
 *
 * Bond with quartic potential (like class2 in lammps).
 *
 */

class BondQuartic;

class BondQuartic : public Bond, public BondQuarticType {
	public:
        BondQuartic(Atom *a, Atom *b, double k2_, double k3_, double k4_, double r0_, int type_=-1);
        BondQuartic(double k2_, double k3_, double k4_, double r0_, int type_=-1); 
        BondQuartic(){};
        int type;
	std::string getInfoString();
};	

void export_BondQuartic();
//end bond quartic classes


//bond fene classes
//
class BondFENEType {
public:
    float k;
    float r0;
    float eps;
    float sig;
    BondFENEType(){};
    bool operator==(const BondFENEType &) const;
    std::string getInfoString();
};
//
//for forcer maps
namespace std {
    template<> struct hash<BondFENEType> {
        size_t operator() (BondFENEType const& bond) const {
            size_t seed = 0;
            boost::hash_combine(seed, bond.k);
            boost::hash_combine(seed, bond.r0);
            boost::hash_combine(seed, bond.eps);
            boost::hash_combine(seed, bond.sig);
            return seed;
        }
    };
}

/*! \brief Bond with a FENE potential
 *
 *
 */





class BondFENE: public Bond, public BondFENEType {
public:
    BondFENE(Atom *a, Atom *b, double k_, double r0_, double eps_, double sig_, int type_=-1);
    BondFENE(double k_, double r0_, double eps_, double sig_, int type_=-1); 
    BondFENE(){};
    int type;
	std::string getInfoString();
};	

void export_BondFENE();
//end bond fene classes



class __align__(16) BondGPU {
    public:
        int myId; //!< ID of this Atom
        int otherId; //!< ID of the other Atom in the Bond
        int type; //!bond type number
        void takeIds(Bond *b); //!copy myId, otherId out of Bond *

};


/*! \typedef Boost Variant for any bond */
typedef boost::variant<
	BondHarmonic, 
	BondQuartic, 
    BondFENE,
	Bond
> BondVariant;

// allows us to extract any type of Bond from a BondVariant
class bondDowncast : public boost::static_visitor<const Bond &> {
    const BondVariant &_bv;
    public:
        bondDowncast(BondVariant &bv) : _bv(bv) {}
        template <typename T>
        const Bond &operator()(const T &b) const {
            return boost::get<T>(_bv);
        }
};
#endif
