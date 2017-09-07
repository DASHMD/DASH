import sys
sys.path = sys.path + ['../../build/python/build/lib.linux-x86_64-2.7' ]
sys.path.append('../../util_py')
from DASH import *
from LAMMPS_Reader import LAMMPS_Reader
import argparse
import re
import matplotlib.pyplot as plt
from math import *
state = State()
state.deviceManager.setDevice(0)
dx = 20
dy = 20
dz = 8
ndim = 3
state.bounds = Bounds(state, lo = Vector(0, -20, -20), hi = Vector(dx*ndim+20, dy*ndim+20, dz*ndim+20))
#state.bounds = Bounds(state, lo = Vector(0, -20, -20), hi = Vector(40, 40, 40))#Vector(dx*ndim+20, dy*ndim+20, dz*ndim+20))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 100

#state.dt = 0.0005
state.dt = 0.0001

ljcut = FixLJCut(state, 'ljcut')
bondHarm = FixBondHarmonic(state, 'bondharm')
angleHarm = FixAngleHarmonic(state, 'angleHarm')
dihedralOPLS = FixDihedralOPLS(state, 'opls')
improperHarm = FixImproperHarmonic(state, 'imp')

tempData = state.dataManager.recordTemperature('all', 100)
state.activateFix(ljcut)
state.activateFix(bondHarm)
state.activateFix(angleHarm)
state.activateFix(dihedralOPLS)
state.activateFix(improperHarm)

unitEng = 0.066
unitLen = 3.5
unitMass = 12
writeconfig = WriteConfig(state, fn='poly_out', writeEvery=100, format='xyz', handle='writer')
writeconfig.unitLen = 1/unitLen
#temp = state.dataManager.recordEnergy('all', 50)
#reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = 12, unitEng = 0.066, bondFix = bondHarm, angleFix = angleHarm, nonbondFix = ljcut, dihedralFix = dihedralOPLS, improperFix=improperHarm, atomTypePrefix = 'PTB7_', setBounds=False)
reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = unitMass, unitEng = unitEng, nonbondFix = ljcut, atomTypePrefix = 'PTB7_', setBounds=False, bondFix = bondHarm,   angleFix = angleHarm, dihedralFix = dihedralOPLS,improperFix=improperHarm,)
reader.read(dataFn = 'poly_min.data')

#1 kelven = 1.38e-23 J/K  / (276/6.022e23) = .00301 temp units
#to tReal * conversion = LJ tempo
#pressure = pReal * unitLen^3/unitEng = 3.5^3/.066
#so to pressure / 649.62 = pReal
tUnit = 0.0301
pUnit = unitLen**3 / unitEng

'''
1 12
2 32.065
3 12
4 19
5 1
6 12
7 16
8 16
9 12
10 1
11 35.453
12 12
13 1
14 126.904
'''



state.atomParams.setValues('PTB7_0', atomicNum=6)
state.atomParams.setValues('PTB7_1', atomicNum=16)
state.atomParams.setValues('PTB7_2', atomicNum=6)
state.atomParams.setValues('PTB7_3', atomicNum=9)
state.atomParams.setValues('PTB7_4', atomicNum=1)
state.atomParams.setValues('PTB7_5', atomicNum=6)
state.atomParams.setValues('PTB7_6', atomicNum=8)
state.atomParams.setValues('PTB7_7', atomicNum=8)
state.atomParams.setValues('PTB7_8', atomicNum=6)
state.atomParams.setValues('PTB7_9', atomicNum=1)
state.atomParams.setValues('PTB7_10', atomicNum=17)
state.atomParams.setValues('PTB7_11', atomicNum=6)
state.atomParams.setValues('PTB7_12', atomicNum=1)
state.atomParams.setValues('PTB7_13', atomicNum=53)

integRelax = IntegratorRelax(state)
integRelax.writeOutput()
#integRelax.run(11, 1e-9)
InitializeAtoms.initTemp(state, 'all', 1)
fixNVT = FixNoseHoover(state, 'temp', 'all', 1, 0.1)
state.activateFix(fixNVT)

#pressureData = state.dataManager.recordPressure('all', 10)
integVerlet = IntegratorVerlet(state)
#integVerlet.run(1500)

state.activateWriteConfig(writeconfig)
state.createMolecule([a.id for a in state.atoms])
print len(state.atoms)
for x in range(ndim):
    for y in range(ndim):
        for z in range(ndim):
            if not (x==0 and y==0 and z==0):
                print x,y,z
                state.duplicateMolecule(state.molecules[0])
                state.molecules[-1].translate(Vector(x*dx, y*dy, z*dz))
#integVerlet.run(2000)
#print temp.vals

#integVerlet = IntegraterVerlet(state)
constPressure = FixPressureBerendsen(state, 'constp', 1*pUnit, .5)
#state.activateFix(constPressure)
ewald = FixChargeEwald(state, "chargeFix", "all")
ewald.setParameters(32, 3.0, 3)
state.activateFix(ewald)

tempData = state.dataManager.recordTemperature('all', 1000)
#print 'energy %f' % (integVerlet.energyAverage('all') * unitEng * len(state.atoms))
integVerlet.run(100)
'''
print state.bounds.hi
print state.bounds.lo
vol = 1.
for i in range(3):
    vol *= state.bounds.hi[i] - state.bounds.lo[i]
print 'vol %f' % vol
print vol*unitLen**3

sumMass = sum([a.m for a in state.atoms])
print 'mass %f' % sumMass
print sumMass * unitMass



#print tempData.vals
#print [p/pUnit for p in pressureData.vals]
print tempData.vals


'''
#print state.atoms[0].pos.dist(state.atoms[1].pos)
#print tempData.vals







