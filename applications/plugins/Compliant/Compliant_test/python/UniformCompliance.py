import Sofa

import sys
import Compliant

from Compliant.Vec import Proxy as vec
from Compliant import Rigid, Tools

import SofaTest

# helper
def insert_point(node, name, position, mass = 1.0):
    res = node.createChild(name)
    
    res.createObject('MechanicalObject', 
                     name = 'dofs',
                     template = 'Vec3d',
                     position = vec(position))

    res.createObject('UniformMass',
                     mass = mass)

    res.createObject('SphereModel', radius = 0.1)
    
    return res

# shared data
class Shared:
    pass

global shared
shared = Shared()

# scene: no gravity, a single particle at (1, 0, 0) with a
# UniformCompliance with random stiffness/damping pulling the particle
# towards the origin

# we check that the force/position are what they should be

def createScene(node):
    scene = Tools.scene( node )

    ode = node.getObject('ode')

    ode.stabilization = "no stabilization"
    ode.warm_start = False

    # ode.debug = True
    # ode.propagate_lambdas = True

    node.gravity = '0 0 0'
    node.dt = 1e-2
    
    num = node.createObject('MinresSolver',
                            name = 'num',
                            iterations = 100,
                            precision = 0)

    # resp = node.createObject('DiagonalResponse')

    script = node.createObject('PythonScriptController',
                               filename = __file__,
                               classname = 'Controller')

    style = node.getObject('style')

    style.displayFlags = 'showCollisionModels'

    # parameters
    mass = 1.0
    stiff = 1e3 
    damping = 0.0

    # dofs
    p = insert_point(scene, 'p', [-1, 0, 0], mass) 
    

    sub = p.createChild('sub')
    sub.createObject('MechanicalObject',
                     name = 'dofs',
                     position = '0 0 0')
    
    sub.createObject('IdentityMapping',
                     template = 'Vec3d,Vec3d')

    compliance = 1/stiff

    sub.createObject('UniformCompliance',
                     template = 'Vec3d',
                     compliance = compliance,
                     damping = damping)
    
    shared.dofs = p.getObject('dofs')

    shared.mass = mass
    shared.stiff = stiff
    shared.damping = damping
    
    return node


class Controller(SofaTest.Controller):


    def onBeginAnimationStep(self, dt):
        self.v = vec(shared.dofs.velocity[0])
        self.q = vec(shared.dofs.position[0])
        
    def onEndAnimationStep(self, dt):

        # wait 1 sec before testing
        if self.root.getTime() < 1:
            return 
        
        # parameters
        m = shared.mass
        k = shared.stiff
        b = shared.damping

        f = -k * self.q
        h = m + (dt * b) + (dt * dt) * k

        rhs = m * self.v + dt * f
        
        # reference solution
        sol = rhs / h

        # precision threshold
        epsilon = 1e-14

        # velocity at time step end computed by solver
        v = vec( shared.dofs.velocity[0] )

        # print v, sol
        error = (v - sol).norm()

        self.should( error < epsilon, 'velocity not what it should be' )
