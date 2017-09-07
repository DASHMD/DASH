import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7', '../build/']
sys.path.append('../util_py')
import matplotlib.pyplot as plt
from LAMMPS_Reader import LAMMPS_Reader
from DASH import *
from math import *
state = State()
state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo = Vector(-10, -10, -10), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 1000
state.grid = AtomGrid(state, 3.6, 3.6, 3.6)

state.dt = 0.0005

ljcut = FixLJCut(state, 'ljcut')
bondHarm = FixBondHarmonic(state, 'bondharm')
angleHarm = FixAngleHarmonic(state, 'angleHarm')
dihedralOPLS = FixDihedralOPLS(state, 'opls')

tempData = state.dataManager.recordTemperature('all', 100)
state.activateFix(ljcut)
state.activateFix(bondHarm)
state.activateFix(angleHarm)
state.activateFix(dihedralOPLS)

unitLen = 3.55
writeconfig = WriteConfig(state, fn='dio_out', writeEvery=10, format='xyz', handle='writer')
writeconfig.unitLen = 1/unitLen
state.activateWriteConfig(writeconfig)

fixNVT = FixNVTRescale(state, 'temp', 'all', [0, 1], [0.1, 0.1], 1)
#state.activateFix(fixNVT)
reader = LAMMPS_Reader(state=state, unitLen = unitLen, unitMass = 12, unitEng = 0.07, bondFix = bondHarm, angleFix = angleHarm, nonbondFix = ljcut, dihedralFix = dihedralOPLS, atomTypePrefix = 'DIO_', setBounds=False)
reader.read(dataFn = 'DIO_VMD.data')

InitializeAtoms.initTemp(state, 'all', 0.1)
state.atomParams.setValues('DIO_0', atomicNum=6)
state.atomParams.setValues('DIO_1', atomicNum=1)
state.atomParams.setValues('DIO_2', atomicNum=53)

print state.atoms[0].pos.dist(state.atoms[1].pos)
integRelax = IntegratorRelax(state)
integRelax.run(100000, 1e-9)
InitializeAtoms.initTemp(state, 'all', 0.1)

integVerlet = IntegratorVerlet(state)
integVerlet.run(100000)
#print state.atoms[0].pos.dist(state.atoms[1].pos)
#print tempData.vals







