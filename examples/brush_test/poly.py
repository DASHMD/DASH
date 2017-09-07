import sys
sys.path = sys.path + ['/home/daniel/Documents/md_engine/core/build/python/build/lib.linux-x86_64-2.7' ]
sys.path.append('/home/daniel/Documents/md_engine/core/util_py')
import matplotlib.pyplot as plt
from LAMMPS_Reader import LAMMPS_Reader
from DASH import *
from math import *
state = State()
state.deviceManager.setDevice(0)
#state.bounds = Bounds(state, lo = Vector(-30, -30, -30), hi = Vector(360, 360, 360))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 10

state.dt = 0.005

ljcut = FixLJCut(state, 'ljcut')
bondFENE = FixBondFENE(state, 'bondFENE')
angleHarm = FixAngleCosineDelta(state, 'angleHarm')
#tempData = state.dataManager.recordTemperature('all', 100)
state.activateFix(ljcut)
state.activateFix(bondFENE)
state.activateFix(angleHarm)

#state.bounds = Bounds(state, Vector(-9, -10, -10), Vector(20, 24, 24))

writeconfig = WriteConfig(state, fn='poly_out', writeEvery=10, format='xyz', handle='writer')
temp = state.dataManager.recordEnergy('all', collectEvery = 50)
reader = LAMMPS_Reader(state=state, unitLen = 1, unitMass = 1, unitEng = 1, bondFix = bondFENE, nonbondFix = ljcut, angleFix = angleHarm, atomTypePrefix = 'POLY_', setBounds=True)


#state.bounds.lo = state.bounds.lo - Vector(0, 0, 3)
#state.bounds.hi = state.bounds.hi + Vector(0, 0, 3)

reader.read(dataFn = 'brush.data', inputFns = ['brush.in', 'brush.init', 'brush.settings'])
print state.bounds.lo
print state.bounds.hi
InitializeAtoms.initTemp(state, 'all', 0.1)

state.atomParams.setValues('POLY_0', atomicNum=6)
state.atomParams.setValues('POLY_1', atomicNum=7)
state.atomParams.setValues('POLY_2', atomicNum=1)
state.activateWriteConfig(writeconfig)

ewald = FixChargeEwald(state, "chargeFix", "all")
ewald.setParameters(32, 3.0, 3)
state.activateFix(ewald)

substrateIds = [a.id for a in state.atoms if a.type == 'POLY_0']
state.createGroup('substrate', substrateIds)
fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='substrate', k=100)
state.activateFix(fixSpring)
#integRelax = IntegratorRelax(state)
#integRelax.writeOutput()
#integRelax.run(11, 1e-9)
fixNVT = FixLangevin(state, 'temp', 'all', .1)
state.activateFix(fixNVT)
zs = [a.pos[2] for a in state.atoms]
print min(zs), max(zs)
print state.bounds.lo[2], state.bounds.hi[2]
print state.bounds.lo
print state.bounds.hi
wallDist = 5
topWall = FixWallHarmonic(state, handle='wall', groupHandle='all', origin=Vector(0, state.bounds.hi[2], 0), forceDir=Vector(0, -1, 0), dist=wallDist, k=100)
bottomWall = FixWallHarmonic(state, handle='wall', groupHandle='all', origin=Vector(0, state.bounds.lo[2], 0), forceDir=Vector(0, 1, 0), dist=wallDist, k=100)



integVerlet = IntegratorVerlet(state)
integVerlet.run(150)
print temp.vals







