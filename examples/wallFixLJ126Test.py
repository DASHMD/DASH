import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
from math import *
from random import random


state = State()
#state.deviceManager.setDevice(0)
state.periodicInterval = 1
state.shoutEvery = 500 #how often is says % done
state.rCut = 2.5 #need to implement padding
state.padding = 0.5
state.seedRNG()

# z bounds taken care of automatically in 2d simulation
state.bounds = Bounds(state, lo=Vector(0, 0, 0),
                             hi=Vector(35,35,35))
state.atomParams.addSpecies(handle='type1', mass=1)


initBoundsType1 = Bounds(state, lo=Vector(5,5,5),
                                hi=Vector(30,30,30))

#InitializeAtoms.populateRand(state,bounds=initBoundsType1,
#                           handle='type1', n=1, distMin = 1.0)
state.addAtom('type1',Vector(16,10,10))
#state.atoms[0].setPos(Vector(50, 10, 10))
state.atoms[0].vel = Vector(10, 0, 0)

#subTemp = 1.0
#InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments

#fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='all', temp=subTemp, timeConstant=0.1)
#state.activateFix(fixNVT)

# Start simulation
#writer = WriteConfig(state, handle='writer', fn='wallFix_test', format='xyz',
#                     writeEvery=2000)
#state.activateWriteConfig(writer)

state.dt = .001


leftWall = FixWallLJ126(state, handle='LJWall',groupHandle='all',
                            origin = Vector(0,0,0),
                            forceDir=Vector(1,0,0),dist=15,sigma=2.4,epsilon=1.0)

rightWall = FixWallLJ126(state,handle='LJWall2',groupHandle='all',
                        origin = Vector(state.bounds.hi[0], 0,0),
                        forceDir=Vector(-1,0,0),dist=15, sigma=2.4, epsilon=1.0)

state.activateFix(leftWall)
state.activateFix(rightWall)
# with a velocity of 10 and box size 35x35x35, only need ~7000 to return to initial position and velocity
# with symmetric LJ walls and sigma = 2.4, eps = 1.0
integrator = IntegratorVerlet(state)
integrator.run(7000)

