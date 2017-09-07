import sys
sys.path.append('../../build/python/build/lib.linux-x86_64-2.7')
from Sim import *
from math import *
import re
import argparse
import matplotlib.pyplot as plt
from random import random

state = State()
state.deviceManager.setDevice(0)
state.periodicInterval = 7
state.shoutEvery = 10000 #how often is says % done
state.is2d = False
state.rCut = 2.5 #need to implement padding
state.padding = 0.5
state.seedRNG()

#20x20x20 simulation box
state.bounds = Bounds(state, lo=Vector(-20.0, -20.0, -20.0), hi=Vector(20.0, 20.0, 20.0))

#set up the interaction parameters
state.atomParams.addSpecies(handle='type1', mass=1, atomicNum=1)

nonBond = FixLJCut(state,handle = 'nonbond')
nonBond.setParameter(param='eps', handleA='type1', handleB='type1', val=1)
nonBond.setParameter(param='sig', handleA='type1', handleB='type1', val=1)
nonBond.setParameter(param='rCut', handleA='type1', handleB='type1', val=pow(2.0,(1.0/6.0))) # purely repulsive LJ

bond = FixBondFENE(state,handle = 'bond')
bond.setBondTypeCoefs(type=0,k=30,r0=1.5,eps=1,sig=1)

angle = FixAngleHarmonic(state,'angle')
angle.setAngleTypeCoefs(type=0,k=30, theta0=pi)
angle.setAngleTypeCoefs(type=1,k=30, theta0=(109.5/180.0)*pi)

#add the macrocycle atoms in a ring in the center of the box
ringNum = 16
increment = (2.0*pi)/ringNum
radius = 1.0/(sqrt(2.0-2.0*cos(increment)))
for i in range(0,ringNum):
	theta = i*increment
	yPos = radius * sin(theta)
	zPos = radius * cos(theta)
	state.addAtom('type1',Vector(0.0,yPos,zPos))

#create the bonds
for i in range(0,ringNum):
	nxtAtom = (i+1)%ringNum
	bond.createBond(state.atoms[i], state.atoms[nxtAtom],type=0)

#create the angles
for i in range(0,ringNum):
	nxtAtom1 = (i+1)%ringNum
	nxtAtom2 = (i+2)%ringNum
	angle.createAngle(state.atoms[i],state.atoms[nxtAtom1],state.atoms[nxtAtom2], type=0)

#register as a molecule
state.createMolecule([i for i in range(0,ringNum)])
state.createGroup('MAC',[i for i in range(0,ringNum)])

# now create the thread
threadNum = 25
offset = -1.0 * (threadNum/2.0 - 0.5)
for i in range(0,threadNum):
	state.addAtom('type1',Vector(offset+float(i), 0.0,0.0))

# add bonds
for i in range(0,threadNum-1):
	bond.createBond(state.atoms[ringNum+i],state.atoms[ringNum+i+1],type=0)

#add angles, for a flexible chain, reduce the angle force constant for this interaction
for i in range(1,threadNum-1):
	angle.createAngle(state.atoms[ringNum+i-1],state.atoms[ringNum+i],state.atoms[ringNum+i+1],type=0)

#register in a group
state.createGroup('thread',[i for i in range(ringNum,ringNum+threadNum)])

#############################################################

# now for the stoppers
armLength = 4
numStopAtms = armLength * 3 + 1
theta = [0,2*pi/3, 4*pi/3]

#stopper at the beginning of the chain
#add the central stopper atoms
state.addAtom('type1',Vector(offset-1.0,0.0,0.0))
stopCenterIdx = ringNum+threadNum
#connect the stopper center to the first atom in the thread
bond.createBond(state.atoms[stopCenterIdx], state.atoms[ringNum],type=0)
#add the angle for the central stopper atom in the thread
angle.createAngle(state.atoms[stopCenterIdx],state.atoms[ringNum],state.atoms[ringNum+1], type=0)
#build the three arms
for i in range(0,3):
	
	#add each atom in the arm
	for j in range(0, armLength):
		r = 1.0+float(j)
		state.addAtom('type1',Vector(offset-1.0,r*sin(theta[i]),r*cos(theta[i])))
	
	#add bonds
	startIdx = stopCenterIdx+i*armLength + 1
	#connect the stopper center to the first atom in the arm
	bond.createBond(state.atoms[stopCenterIdx], state.atoms[startIdx],type=0)
	#connect up the rest of the stopper arm
	for j in range(0, armLength-1):
		bond.createBond(state.atoms[startIdx+j], state.atoms[startIdx+j+1],type=0)
		
	#add angles, stopper should be very stiff
	#add angle between thread and stopper arm
	angle.createAngle(state.atoms[ringNum],state.atoms[stopCenterIdx],state.atoms[startIdx],type=1)
	#add angle between stopper arm and central stopper atom (need armLength >= 2)
	angle.createAngle(state.atoms[stopCenterIdx],state.atoms[startIdx],state.atoms[startIdx+1],type=0)
	#now add the angles for the arms
	for j in range(1,armLength-1):
		angle.createAngle(state.atoms[startIdx+j-1],state.atoms[startIdx+j],state.atoms[startIdx+j+1],type=0)

#add the inter-arm angle potentials
startIdx = stopCenterIdx + 1
angle.createAngle(state.atoms[startIdx],state.atoms[stopCenterIdx],state.atoms[startIdx+armLength],type=1)
angle.createAngle(state.atoms[startIdx],state.atoms[stopCenterIdx],state.atoms[startIdx+2*armLength],type=1)
angle.createAngle(state.atoms[startIdx+armLength],state.atoms[stopCenterIdx],state.atoms[startIdx+2*armLength],type=1)

#add the beginning stopper to a group
state.createGroup('stop1',[i for i in range(ringNum+threadNum,ringNum+threadNum+numStopAtms)])

############################################################################

#stopper at the beginning of the chain
#add the central stopper atoms
state.addAtom('type1',Vector(-1.0*offset+1.0,0.0,0.0))
stopCenterIdx = ringNum+threadNum+numStopAtms
lastThreadIdx = ringNum+threadNum-1
#connect the stopper center to the first atom in the thread
bond.createBond(state.atoms[stopCenterIdx], state.atoms[lastThreadIdx],type=0)
#add the angle for the central stopper atom in the thread
angle.createAngle(state.atoms[stopCenterIdx],state.atoms[lastThreadIdx],state.atoms[lastThreadIdx-1], type=0)
#build the three arms
for i in range(0,3):
	
	#add each atom in the arm
	for j in range(0, armLength):
		r = 1.0+float(j)
		state.addAtom('type1',Vector(-1.0*offset+1.0,r*sin(theta[i]),r*cos(theta[i])))
	
	#add bonds
	startIdx = stopCenterIdx+i*armLength + 1
	#connect the stopper center to the first atom in the arm
	bond.createBond(state.atoms[stopCenterIdx], state.atoms[startIdx],type=0)
	#connect up the rest of the stopper arm
	for j in range(0, armLength-1):
		bond.createBond(state.atoms[startIdx+j], state.atoms[startIdx+j+1],type=0)
		
	#add angles, stopper should be very stiff
	#add angle between thread and stopper arm
	angle.createAngle(state.atoms[lastThreadIdx],state.atoms[stopCenterIdx],state.atoms[startIdx],type=1)
	#add angle between stopper arm and central stopper atom (need armLength >= 2)
	angle.createAngle(state.atoms[stopCenterIdx],state.atoms[startIdx],state.atoms[startIdx+1],type=0)
	#now add the angles for the arms
	for j in range(1,armLength-1):
		angle.createAngle(state.atoms[startIdx+j-1],state.atoms[startIdx+j],state.atoms[startIdx+j+1],type=0)

#add the inter-arm angle potentials
startIdx = stopCenterIdx + 1
angle.createAngle(state.atoms[startIdx],state.atoms[stopCenterIdx],state.atoms[startIdx+armLength],type=1)
angle.createAngle(state.atoms[startIdx],state.atoms[stopCenterIdx],state.atoms[startIdx+2*armLength],type=1)
angle.createAngle(state.atoms[startIdx+armLength],state.atoms[stopCenterIdx],state.atoms[startIdx+2*armLength],type=1)

#add the end stopper to a group
state.createGroup('stop2',[i for i in range(ringNum+threadNum+numStopAtms,ringNum+threadNum+2*numStopAtms)])


##########################################################

state.createMolecule([i for i in range(ringNum,len(state.atoms))])
state.createGroup('poly',[i for i in range(ringNum,len(state.atoms))])

#activate the interactions
state.activateFix(nonBond)
state.activateFix(bond)
state.activateFix(angle)

InitializeAtoms.initTemp(state, 'all', 1.0)
	
#Langevin dynamics
fixNVT = FixLangevin(state, 'temp', 'all', 1.0)
fixNVT.setParameters(gamma = 1.0)
state.activateFix(fixNVT)
state.dt = 0.002

#set up output
writeconfig = WriteConfig(state, fn='2rotax_unwrap', writeEvery=100, format='xyz', handle='writer', groupHandle='all',unwrapMolecules=True)
state.activateWriteConfig(writeconfig)
ener = state.dataManager.recordEnergy(handle='all', mode='scalar', interval = 1000)
temp = state.dataManager.recordTemperature(interval = 1000)

#run MD
integVerlet = IntegratorVerlet(state)
integVerlet.run(1000)

	
	
	
	
	
	
	
	
