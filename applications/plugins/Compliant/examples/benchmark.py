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

    # time step
    node.dt = 0.005

    # scene node
    scene = Tools.scene( node )

    # display flags
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
    ode = node.getObject('ode',)
    ode.stabilization = True
    ode.stabilization_damping = 0

    # main solver
    num = node.createObject('BenchmarkSolver', name = 'num')

    response = node.createObject('LDLTResponse', name = 'response')
    
    iterations = 300
    precision = 1e-8

    
    # we need compliantdev for qpsolver
    node.createObject('RequiredPlugin', 
                      pluginName = 'CompliantDev')

    # benchmarks
    shared.pgs = node.createObject('Benchmark', name = 'bench-pgs')
    shared.qp = node.createObject('Benchmark', name = 'bench-qp')

    # solvers
    pgs = node.createObject('SequentialSolver',
                            name = 'pgs',
                            iterations = iterations,
                            precision = precision,
                            bench = '@./bench-pgs')

    qp = node.createObject('QPSolver',
                           name = 'qp',
                           iterations = iterations,
                           precision = precision,
                           bench = '@./bench-qp',
                           schur = True)


    # plane
    plane = Rigid.Body('plane')
    plane.visual = dir + '/mesh/ground.obj'
    plane.collision = plane.visual
    plane.mass_from_mesh( plane.visual, 10 )
    plane.node = plane.insert( scene )
    
    plane.node.createObject('FixedConstraint', 
                             indices = '0')

    # boxes
    n_boxes = 10

    for i in xrange(n_boxes):
        box = Rigid.Body('box-{0}'.format(i))
        box.visual = dir + '/mesh/cube.obj'
        box.collision = box.visual
        box.dofs.translation = [0, 2.5 * (i + 1), 0]
        box.mass_from_mesh( box.visual, 50 )
        box.node = box.insert( scene )

    
from itertools import izip
from matplotlib import pyplot as plt
import matplotlib.animation as anim
import numpy as np

# scene controller
class Controller(Sofa.PythonScriptController):
     
    def onLoaded(self,node):
        self.node = node

        plt.ion()
        plt.show()

        return 0
          
    def reset(self):
        return 0
          
    def onBeginAnimationStep(self, dt):
        return 0

    def plot_bench(self):
        plt.clf()

        for bench in [shared.pgs, shared.qp]:
            values = []
            for (p, d, c, o) in izip(bench.primal, 
                                     bench.dual,
                                     bench.complementarity,
                                     bench.optimality):
                values.append(p[0] + d[0] + c[0] + o[0])
                
            # total time
            time = [x[0] for x in bench.duration]
            timesum = (bench.factor + np.cumsum( time )) / 1000

            if len(time) > 0:
                plt.plot(timesum, values, label = bench.name)

        # self.ax.set_yscale('log')
        plt.yscale('log')
        plt.draw()

    def report_bench( self, bench ):
        # display the values from the bench object
        total = []
        
        for (p, d, c, o) in izip(bench.primal, 
                                 bench.dual,
                                 bench.complementarity,
                                 bench.optimality):
            total.append(p[0] + d[0] + c[0] + o[0])

        time = [x[0] for x in bench.duration]
        timesum = (bench.factor + np.cumsum( time )) / 1000

        print bench.name
        print 'factor:', bench.factor / 1000, 'ms'
        print 'convergence:', total
        print 'duration (ms):', timesum
        print 


    def detailed_report_bench( self, bench ):
        # display the values from the bench object
        total = []
        
        time = [x[0] for x in bench.duration]
        timesum = (bench.factor + np.cumsum( time )) / 1000

        print bench.name
        print 'factor:', bench.factor / 1000, 'ms'

        print 'primal:', [ x[0] for x in bench.primal ]
        print 'dual:', [ x[0] for x in bench.dual ]
        print 'complementarity:', [ x[0] for x in bench.complementarity ]
        print 'convergence:', total
        print 'duration (ms):', timesum
        print 

        
    def report(self):
        # print benchmark report
        print 't =', self.node.getTime()
        print

        for bench in [shared.pgs, shared.qp]:
            self.print_report_bench( bench )
            

    def onEndAnimationStep(self, dt):
        self.plot_bench()
        # self.print_report()
        # self.detailed_report_bench( shared.qp )
        return 0
          
    def bwdInitGraph(self,node):
        return 0
