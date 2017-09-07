import sys
sys.path = sys.path + ['../build/python/build/lib.linux-x86_64-2.7']
#from DASH import *
from DASH import *
state = State()
state.deviceManager.setDevice(0)
state.bounds = Bounds(state, lo = Vector(0, 0, 0), hi = Vector(55.12934875488, 55.12934875488, 55.12934875488))
state.rCut = 3.0
state.padding = 0.6
state.periodicInterval = 7
state.shoutEvery = 100

state.atomParams.addSpecies(handle='spc1', mass=1, atomicNum=1)
nonbond = FixLJCut(state, 'cut')
nonbond.setParameter('sig', 'spc1', 'spc1', 1)
nonbond.setParameter('eps', 'spc1', 'spc1', 1)
state.activateFix(nonbond)

f = open('init.xml').readlines()
for i in range(len(f)):
    bits = [float(x) for x in f[i].split()]
    state.addAtom('spc1', Vector(bits[0], bits[1], bits[2]))

#state.addAtom('spc1', pos = Vector(10, 10, 10))
#state.addAtom('spc1', pos = Vector(10.5, 10.5, 10.7))
InitializeAtoms.initTemp(state, 'all', 1.2)

fixNVT = FixLangevin(state, 'temp', 'all', 1.2)
#fixNVT = FixNVTRescale(state, 'temp', 'all', 1.2)
#fixNPT = FixNoseHoover(state,'npt','all')
#fixNPT.setTemperature(1.2,5.0*state.dt)
#fixNPT.setPressure('ANISO',0.2,1000*state.dt)
state.activateFix(fixNVT)

integVerlet = IntegratorVerlet(state)

#empData = state.dataManager.recordTemperature('all','scalar', 100)
#pressureData = state.dataManager.recordPressure('all','scalar', 1)
#engData = state.dataManager.recordEnergy('all', 100)
#boundsData = state.dataManager.recordBounds(100)

#pressure = FixPressureBerendsen(state, "constP", .2, 10, 1);
#state.activateFix(pressure);
#deform = FixDeform(state, 'def', 'all', 1, Vector(1, 0, 0))
#state.activateFix(deform)

writeconfig = WriteConfig(state, fn='test_out', writeEvery=200, format='xyz', handle='writer')
#state.activateWriteConfig(writeconfig)
#state.tuneEvery = 10000
integVerlet.run(10000)
sumV = 0.
for a in state.atoms:
    sumV += a.vel.lenSqr()
#print engData.vals
#print sumV / len(state.atoms)/3.0
#plt.plot(pressureData.turns, pressureData.vals)
#plt.show()
#plt.show()
#state.dataManager.stopRecord(tempData)
#integVerlet.run(10000)
#print len(tempData.vals)
#plt.plot([x for x in engData.vals])
#plt.show()
#print sum(tempData.vals) / len(tempData.vals)
#print boundsData.vals[0].getSide(1)
#print engData.turns[-1]
#print 'last eng %f' % engData.vals[-1]
#print state.turn
#print integVerlet.energyAverage('all')
#perParticle = integVerlet.energyPerParticle()
#print sum(perParticle) / len(perParticle)
