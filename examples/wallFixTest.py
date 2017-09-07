import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
from DASH import *
from math import *
from random import random


state = State()
#state.deviceManager.setDevice(0)
state.periodicInterval = 1
state.shoutEvery = 1000 #how often is says % done
state.rCut = 2.5 #need to implement padding
state.padding = 0.5
state.seedRNG()

# z bounds taken care of automatically in 2d simulation
state.bounds = Bounds(state, lo=Vector(0, 0, 0),
                             hi=Vector(60, 60, 60))
state.grid = AtomGrid(state, dx=3.5, dy=3.5, dz=3.5) #as is dz
state.atomParams.addSpecies(handle='type1', mass=1)


initBoundsType1 = Bounds(state, lo=Vector(5,5,0),
                                hi=Vector(15,10,20))

InitializeAtoms.populateRand(state,bounds=initBoundsType1,
                            handle='type1', n=5, distMin = 1.0)


subTemp = 1.0
InitializeAtoms.initTemp(state, 'all', subTemp) #need to add keyword arguments

fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='all', temp=subTemp, timeConstant=0.1)
state.activateFix(fixNVT)

# Start simulation
#writer = WriteConfig(state, handle='writer', fn='wallFix_test', format='xyz',
#                     writeEvery=2000)
#state.activateWriteConfig(writer)

state.dt = 0.001

topWall = FixWallHarmonic_temp(state, handle='harmonicWall',groupHandle='all',
                            origin = Vector(0,0,0),
                            forceDir=Vector(1,0,0),dist=29,k=2.257)

state.activateFix(topWall)

integrator = IntegratorVerlet(state)
integrator.run(5000)

