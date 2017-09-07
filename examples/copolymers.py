from DASH import *
from random import random
from math import *

state = State()
state.deviceManager.setDevice(0)
state.periodicInterval = 10
state.rCut = 0.082
state.padding=0.07

bondlen=1.0/sqrt(31.0)
chiN=37.0
kappaN=50.0
boxsize=1.78

n_poly=int(boxsize*boxsize*boxsize*4096/32)
print "number of chains ",n_poly
r_int=0.082

normc = 1.0/32.0


bnds=Bounds(state, lo=Vector(-0.5*boxsize, -0.5*boxsize, -0.5*boxsize), hi=Vector( 0.5*boxsize,  0.5*boxsize, 0.5*boxsize))
state.bounds = bnds
state.grid = AtomGrid(state, dx=0.178, dy=0.178, dz=0.178)
state.atomParams.addSpecies(handle='A', mass=1)
state.atomParams.addSpecies(handle='B', mass=1)

total_beads=n_poly*32
filename="initial_L_1.78_rho_4096_pol_16_16.dat"
ifile = open(filename, 'r') 

print "reading file ", filename
for i in range(total_beads):
    line= ifile.readline()
    words = line.split()
    state.addAtom(handle=words[3],pos=Vector(float(words[0]),float(words[1]),float(words[2])))


bonds=FixBondHarmonic(state,handle="bond")
for line in ifile:
    words = line.split()
    bonds.createBond(state.atoms[int(words[0])], state.atoms[int(words[1])], 3.0/pow(bondlen,2.0), 0.0)
ifile.close() 
state.activateFix(bonds)

ticg = FixTICG(state, handle='ljcut')

state.activateFix(ticg)

ticg.setParameter('C', 'A','A', normc*kappaN)
ticg.setParameter('rCut', 'A','A', r_int)
ticg.setParameter('C', 'B','B', normc*kappaN)
ticg.setParameter('rCut', 'B','B', r_int)
ticg.setParameter('C', 'A','B', normc*(kappaN+chiN))
ticg.setParameter('rCut', 'A','B', r_int)

writer = WriteConfig(state, handle='writer', fn='diblock_test_*', format='xyz', writeEvery=1000) 
state.activateWriteConfig(writer)
state.shoutEvery=1000
state.dt=0.0003

integrator = IntegratorLangevin(state,1.0)
integrator.set_params(1,1.0)
integrator.run(100000)
