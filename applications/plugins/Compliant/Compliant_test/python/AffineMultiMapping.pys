import Sofa

import sys

import Compliant

from Compliant.Vec import Proxy as vec
from Compliant import Rigid, Tools

import SofaTest


# shared data
class Shared:
    pass

global shared
shared = Shared()

# helper TODO move this to Compliant API
def insert_point(node, name, position, mass = 1.0):
    res = node.createChild(name)
    
    res.createObject('MechanicalObject', 
                     name = 'dofs',
                     template = 'Vec3d',
                     position = vec(position))

    res.createObject('UniformMass',
                     mass = mass)

    res.createObject('SphereModel', radius = 0.01)
    
    return res


# 2 particles, we force the alpha-interpolation to lie at the origin,
# using AffineMultiMapping, check result after some time.

def createScene(node):
    scene = Tools.scene( node )

    ode = node.getObject('ode')

    ode.stabilization = "no stabilization"
    # ode.warm_start = False
    # ode.debug = True


    node.gravity = '0 0 0'
    node.dt = 1e-2
    
    num = node.createObject('MinresSolver',
                            name = 'num',
                            iterations = 10,
                            precision = 0)

    style = node.getObject('style')
    style.displayFlags = 'showCollisionModels showBehaviorModels'

    script = node.createObject('PythonScriptController',
                               filename = __file__,
                               classname = 'Controller')

    mass = 1
    
    p1 = insert_point(scene, 'p1', [-1, 1, 0], mass)
    p2 = insert_point(scene, 'p2', [3, 1, 0], mass)

    out = scene.createChild('out')
    out.createObject('MechanicalObject', 
                     name = 'dofs',
                     template = 'Vec1d', 
                     position = '0 0 0')

    alpha = 0.2
    shared.alpha = alpha

    matrix = vec([1 - alpha, 0, 0, alpha, 0, 0,
                  0, 1 - alpha, 0, 0, alpha, 0,
                  0, 0, 1 - alpha, 0, 0, alpha] )
    
    value = vec([0, 0,  0])
    
    out.createObject('AffineMultiMapping',
                     template = 'Vec3d,Vec1d',
                     input = '@../p1/dofs @../p2/dofs',
                     output = '@dofs',
                     matrix = str(matrix),
                     value = str(value))
    
    out.createObject('UniformCompliance',
                     template = 'Vec1d',
                     compliance = 1e-5)

    shared.p1 = p1.getObject('dofs')
    shared.p2 = p2.getObject('dofs')


class Controller(SofaTest.Controller):

    def onEndAnimationStep(self, dt):
        if self.node.getTime() > 0.5:

            p1 = vec( shared.p1.position[0] )
            p2 = vec( shared.p2.position[0] )
        
            value = (1 - shared.alpha) * p1 + shared.alpha * p2

            self.should( value.norm() == 0 )
        
