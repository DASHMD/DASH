from math import *


def create_TIP3P(state, oxygenHandle, hydrogenHandle, center=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=-0.8340)
    h1Pos = center + Vector(0.9572, 0, 0)
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.4170)
    theta = 1.824218134
    h2Pos = center + Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.4170)
    return state.createMolecule([state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])


def create_TIP3P_long(state, oxygenHandle, hydrogenHandle, center=None):
    if center==None:
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=-0.83)
    h1Pos = center + Vector(0.9572, 0, 0)
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.415)
    theta = 1.824218134
    h2Pos = center + Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.4170)
    return state.createMolecule([state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

def create_TIP4P(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None):
    if center==None
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=0)
    offset1 = Vector(0.9572, 0, 0)
    h1Pos = center + offset1
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.52)
    theta = 1.824218134
    offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    h2Pos = center + offset2
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.52)
    mSiteOffset = (offset1 + offset2).normalized() * 0.15

    state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.04)
    return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

def create_TIP4P_long(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None):
    if center==None
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=0)
    offset1 = Vector(0.9572, 0, 0)
    h1Pos = center + offset1
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.5242)
    theta = 1.824218134
    offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    h2Pos = center + offset2
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.5242)
    mSiteOffset = (offset1 + offset2).normalized() * 0.15

    state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.0484)
    return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])

def create_TIP4P_2005(state, oxygenHandle, hydrogenHandle, mSiteHandle, center=None):
    if center==None
        center = state.Vector(0, 0, 0)
    state.addAtom(handle=oxygenHandle, pos=center, q=0)
    offset1 = Vector(0.9572, 0, 0)
    h1Pos = center + offset1
    state.addAtom(handle=hydrogenHandle, pos=h1Pos, q=0.5897)
    theta = 1.824218134
    offset2 = Vector(cos(theta)*0.9572, sin(theta)*0.9572, 0)
    h2Pos = center + offset2
    state.addAtom(handle=hydrogenHandle, pos=h2Pos, q=0.5897)
    mSiteOffset = (offset1 + offset2).normalized() * 0.15

    state.addAtom(handle=mSiteHandle, pos=center+mSiteOffset, q=-1.1794)
    return state.createMolecule([state.atoms[-4].id, state.atoms[-3].id, state.atoms[-2].id, state.atoms[-1].id])


