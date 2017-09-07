import re
import sys
import math
DEGREES_TO_RADIANS = math.pi / 180.

print 'WARNING: THE NAMD READER IS NOT COMPLETE'

class NAMD_Bonded_Forcer:
    def __init__(self, type, atomTypes):
        self.type = type
#these are types with the atomTypePrefix
        self.atomTypes = atomTypes
    def tryOrder(self, atomTypesForcer, atomTypes):
        matchLevel = 3
        for i in range(len(atomTypesForcer)):
            atomType = atomTypesForcer[i]
            if atomType[-1] == 'x':
                if matchLevel > 1:
                    matchLevel = 1
            elif atomType[-1] == 'X':
                if atomType[:-1] == atomTypes[i][:len(atomType)-1]:
                    if matchLevel > 2:
                        matchLevel = 2
                else:
                    matchLevel = 0
            elif atomType != atomTypes[i]:
                matchLevel = 0
       # print self.atomTypes, atomTypes, matchLevel
        return matchLevel
    def matchLevel(self, atomTypes):
#0 -> no match, 1-> total wildcard, 2->atom + X, 3->perfect
        matchOrder1 = self.tryOrder(self.atomTypes, atomTypes)
        self.atomTypes.reverse()
        matchOrder2 = self.tryOrder(self.atomTypes, atomTypes)
        return max(matchOrder1, matchOrder2)

#set to wildcard



class NAMD_Reader:
    def __init__(self, state=None, nonbondFix=None, bondFix=None, angleFix=None, dihedralFix=None, improperFix=None, atomTypePrefix = '', setBounds=True):
        assert(state != None)
        self.state = state
        self.nonbondFix = nonbondFix
        self.bondFix = bondFix
        self.angleFix = angleFix
        self.dihedralFix = dihedralFix
        self.improperFix = improperFix
        self.myAtomTypeIds = []
        self.myAtomHandles = []
        self.atomTypePrefix = atomTypePrefix
        self.setBounds = setBounds

        self.NAMDBondTypes = []
        self.NAMDAngleTypes = []
        self.NAMDDihedralTypes = []
        self.NAMDImproperTypes = []

        self.namdToSimId = {}

    def read(self, inputFn='', structureFn='', coordinatesFn='', parametersFn=''):
        self.inputFile = open(inputFn, 'r')
        self.structureFile = open(structureFn, 'r')
        self.structureFileLines = self.structureFile.readlines()
        self.coordinateFile = open(coordinatesFn, 'r')
        self.coordinateFileLines = self.coordinateFile.readlines()
        self.parameterFile = open(parametersFn, 'r')
        self.parameterFileLines = self.parameterFile.readlines()


        self.readAtoms()
        #self.readAtomTypes()


        #might not want to change bounds if you just adding parameters to an existing simulation

        if self.setBounds:
            self.readBounds()
        if self.nonbondFix != None:
            self.readPairCoefs()


        if self.bondFix != None:
            self.readBondCoefs()
            self.readBonds()
            print 'Read bonds'

        if self.angleFix != None:
            self.readAngleCoefs()
            self.readAngles()
            print 'Read angles'
        if self.dihedralFix != None:
            self.readDihedralCoefs()
            self.readDihedrals()
            print 'Read dihedrals'
      #  if self.improperFix != None:
      #      self.readImpropers()
      #      self.readImproperCoefs()




    def readAtoms(self):
        nAtoms = 0
        idxStart = 0
        for i in xrange(len(self.structureFileLines)):
            if 'NATOM' in self.structureFileLines[i]:
                nAtoms = int(self.structureFileLines[i].split()[0])
                idxStart = i+1
                break
        for i in range(idxStart, idxStart + nAtoms):
            bits = self.structureFileLines[i].split()
            atomType = self.atomTypePrefix + bits[5]
            mass = float(bits[7])
            q = float(bits[6])
            self.state.atomParams.addSpecies(atomType, mass)
            atomId = self.state.addAtom(atomType, self.state.Vector(), q)
            namdId = int(bits[0])
            self.namdToSimId[namdId] = atomId
        for i in range(1, nAtoms+1):
#deposit.rcsb.org/adit/docs/pdb_atom_format.html
            line = self.coordinateFileLines[i]
            self.state.atoms[i-1].pos[0] = float(line[31:39])
            self.state.atoms[i-1].pos[1] = float(line[39:47])
            self.state.atoms[i-1].pos[2] = float(line[47:55])

    def readInMultiAtom(self, tag, forcerTypes, nAtomsPerForcer, createMember):
        nItems = 0
        idxStart = 0
        for i in xrange(len(self.structureFileLines)):
            if tag in self.structureFileLines[i]:
                nItems = int(self.structureFileLines[i].split()[0])
                idxStart = i+1
        allEntries = []
        i = idxStart
        bits = self.structureFileLines[i].split()
        while len(bits):
            allEntries += bits
            i+=1
            bits = self.structureFileLines[i].split()

        for i in range(0, len(allEntries), nAtomsPerForcer):
            ids = []
            types = []
            for j in range(nAtomsPerForcer):
                ids.append(self.namdToSimId[int(allEntries[i+j])])
                types.append(self.atomFromId(ids[-1]).type)
            forcer = self.pickBestForcer(types, forcerTypes)
            atoms = [self.state.atoms[self.state.idToIdx(id)] for id in ids]
            createMember(atoms, forcer.type)

    def readBonds(self):
        def createBond(atoms, type):
            self.bondFix.createBond(atoms[0], atoms[1], type=type)
        self.readInMultiAtom('NBOND', self.bondTypes, 2, createBond)
    def readAngles(self):
        def createAngle(atoms, type):
            self.angleFix.createAngle(atoms[0], atoms[1], atoms[2], type=type)
        self.readInMultiAtom('NTHETA', self.angleTypes, 3, createAngle)
    def readDihedrals(self):
        def createDihedral(atoms, type):
            self.dihedralFix.createDihedral(atoms[0], atoms[1], atoms[2], atoms[3], type=type)
        self.readInMultiAtom('NPHI', self.dihedralTypes, 4, createDihedral)

    def atomFromId(self, id):
        idx = self.state.idToIdx(id)
        return self.state.atoms[idx]
    def readPairCoefs(self):
        for i in range(len(self.parameterFileLines)):
            if 'NONBONDED' in self.stripComments(self.parameterFileLines[i]):
                break
        i+=2

        while i < len(self.parameterFileLines) and self.parameterFileLines[i] != '':
            bits = self.stripComments(self.parameterFileLines[i]).split()
            if len(bits):
                handle = self.atomTypePrefix + bits[0]
                epsInput = float(bits[2])
                rMinInput = float(bits[3])
                eps = -epsInput
                sigma = 2*(rMinInput) / pow(2.0, 1.0 / 6.0) #IS THE X2 CORRECT?  ASK AMIN
                print sigma
               # sigma = (rMinInput) / pow(2.0, 1.0 / 6.0)#IS THE X2 CORRECT?  ASK AMIN
                self.nonbondFix.setParameter('sig', handle, handle, sigma)
                self.nonbondFix.setParameter('eps', handle, handle, eps)
            i+=1



    def readBondCoefs(self):
        for i in range(len(self.parameterFileLines)):
            if 'BOND' in self.stripComments(self.parameterFileLines[i]):
                break
        i+=2
        self.bondTypes = []
        while i < len(self.parameterFileLines) and len(self.parameterFileLines[i].split()) > 0:
            bits = self.stripComments(self.parameterFileLines[i]).split()
            if len(bits):
                atomTypes = [self.atomTypePrefix + b for b in bits[:2]]

                k = 2 * float(bits[2])
                r0 = float(bits[3])
                type = len(self.bondTypes)
                self.bondFix.setBondTypeCoefs(type, k, r0)
                self.bondTypes.append(NAMD_Bonded_Forcer(type, atomTypes))

            i+=1

    def readAngleCoefs(self):
        for i in range(len(self.parameterFileLines)):
            if 'ANGLE' in self.stripComments(self.parameterFileLines[i]):
                break
        i+=2
        self.angleTypes = []
        while i < len(self.parameterFileLines) and len(self.parameterFileLines[i].split()) > 0:
            bits = self.stripComments(self.parameterFileLines[i]).split()
            if len(bits):
                atomTypes = [self.atomTypePrefix + b for b in bits[:3]]
                k = float(bits[3]) * 2 #2 because LAMMPS includes the 1/2 in its k

                theta0 = float(bits[4]) * DEGREES_TO_RADIANS
                type = len(self.angleTypes)
                self.angleFix.setAngleTypeCoefs(type, k, theta0)
                self.angleTypes.append(NAMD_Bonded_Forcer(type, atomTypes))

            i+=1

#x CA CD x 2.65 2 180.
    def readDihedralCoefs(self):
        for i in range(len(self.parameterFileLines)):
            if 'DIHEDRAL' in self.stripComments(self.parameterFileLines[i]):
                break
        i+=2
        self.dihedralTypes = []
        while i < len(self.parameterFileLines) and len(self.parameterFileLines[i].split()) > 0:
            bits = self.stripComments(self.parameterFileLines[i]).split()
            if len(bits):
                atomTypes = [self.atomTypePrefix + b for b in bits[:4]]
                k = float(bits[4]) #2 because LAMMPS includes the 1/2 in its k

                n = int(bits[5])
                d = float(bits[6]) * DEGREES_TO_RADIANS
                type = len(self.dihedralTypes)
                self.dihedralFix.setDihedralTypeCoefs(type, k, n, d)
                self.dihedralTypes.append(NAMD_Bonded_Forcer(type, atomTypes))

            i+=1
    def isNums(self, bits):
        for b in bits:
            if b[0] == '#':
                break
            try:
                float(b)
            except ValueError:
                return False
        return len(bits)
    def stripComments(self, line):
        if '!' in line:
            return line[:line.index('!')]
        return line
    def emptyLine(self, line):
        bits = line.split()
        return len(bits)==0 or bits[0][0]=='!'
    def emptyLineSplit(self, bits):
        return len(bits)==0 or bits[0][0]=='!'

    def pickBestForcer(self, types, forcers):
        fits = [f.matchLevel(types) for f in forcers]
        if max(fits) == 0:
            print fits
            print types
            print 'COULD NOT FIT FORCER TO TYPE'
            assert(max(fits))
        return forcers[fits.index(max(fits))]
