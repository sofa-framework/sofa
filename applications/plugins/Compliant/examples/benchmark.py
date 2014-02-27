import Sofa

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + '/..') # wtf ?

import Plugin

import math

from Compliant import Rigid, Vec, Quaternion, Tools, Control

class Shared:
    pass

global shared
shared = Shared()

dir = Tools.path( __file__ )

def createScene(node):

    # controller
    node.createObject('PythonScriptController', 
                      filename = __file__,
                      classname = 'Controller' )

    node.dt = 0.005

    # friction coefficient
    shared.mu = 0.5

    scene = Tools.scene( node )

    style = node.getObject('style')
    style.findData('displayFlags').showMappings = True

    # collision detection
    proximity = node.getObject('proximity')
    proximity.alarmDistance = 0.5
    proximity.contactDistance = 0.1

    # contat manager
    manager = node.getObject('manager')
    manager.response = 'CompliantContact'
    manager.responseParams = 'compliance=0'
    
    # integrator
    ode = node.getObject('ode')
    ode.stabilization = True

    # benchmark
    bench = node.createObject('Benchmark', name = 'bench')
    shared.bench = bench
    
    # numerical solver
    num = node.createObject('SequentialSolver',
                            name = 'num',
                            iterations = 100,
                            precision = 1e-8,
                            bench = '@./bench')
    


    # plane
    plane = Rigid.Body('plane')
    plane.visual = dir + '/mesh/ground.obj'
    plane.collision = plane.visual
    plane.mass_from_mesh( plane.visual, 10 )
    plane.node = plane.insert( scene )
    
    plane.node.createObject('FixedConstraint', 
                             indices = '0')

    # box
    box = Rigid.Body('box')
    box.visual = dir + '/mesh/cube.obj'
    box.collision = box.visual
    box.dofs.translation = [0, 3, 0]
    box.mass_from_mesh( box.visual, 50 )
    box.node = box.insert( scene )

    
from itertools import izip


# scene controller
class Controller(Sofa.PythonScriptController):
     
    def onLoaded(self,node):
        return 0
          
    def reset(self):
        return 0
          
    def onBeginAnimationStep(self, dt):
        return 0

    def onEndAnimationStep(self, dt):

        # display the values from the bench object
        total = []
        
        for (p, d, c) in izip(shared.bench.primal, 
                              shared.bench.dual,
                              shared.bench.complementarity):

            total.append(p[0] + d[0] + c[0])

        print total

        return 0

          
    def bwdInitGraph(self,node):
        return 0
