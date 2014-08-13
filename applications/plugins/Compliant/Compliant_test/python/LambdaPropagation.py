import Sofa

import sys

import Compliant
path = Compliant.path()


from Compliant.Vec import Proxy as vec

from Compliant import Rigid, Tools


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


# scene: gravity = 0, -1, 0. we let two masses (m = 1) fall on a fixed
# rigid body (m = 1), and check constaint forces acting on the rigid
# constraint and body after collision.

# the constraint force on the rigid fixed constraint should be 3,
# while the net constraint forces acting on the rigid should be -gravity = 1

def createScene(node):
    scene = Tools.scene( node )

    ode = node.getObject('ode')

    ode.stabilization = "pre-stabilization"
    ode.warm_start = False
    ode.propagate_lambdas = True
    
    # ode.debug = True

    node.gravity = '0 -1 0'
    
    num = node.createObject('SequentialSolver',
                            name = 'num',
                            iterations = 200,
                            precision = 0)

    resp = node.createObject('DiagonalResponse')

    manager = node.getObject('manager')
    manager.response = 'CompliantContact'

    script = node.createObject('PythonScriptController',
                               filename = __file__,
                               classname = 'Controller')

    style = node.getObject('style')

    style.displayFlags = 'showBehaviorModels showCollisionModels'
    
    proximity = node.getObject('proximity')
    proximity.contactDistance = 0.01
    
    # dofs
    p1 = insert_point(scene, 'p1', [-1, 1, 0]) 
    p2 = insert_point(scene, 'p2', [1, 1, 0]) 

    rigid = Rigid.Body('rigid')
    rigid.collision = path + '/examples/mesh/ground.obj'

    rigid.node = rigid.insert( scene )

    ground = Rigid.Body('ground')
    ground.node = ground.insert( scene )
    ground.node.createObject('FixedConstraint', indices = '0')
    
    # blocked joint between ground/rigid
    joint = Rigid.Joint('joint')
    joint.absolute(Rigid.Frame(), ground.node, rigid.node)
    
    joint.node = joint.insert( scene )
    
    shared.joint = joint
    shared.body = rigid
    
    return node


class Controller(Sofa.PythonScriptController):
    
    def onLoaded(self, node):
        self.node = node

    def onEndAnimationStep(self, dt):

        # wait for simulation to settle
        if self.node.getRoot().getTime() > 1: 
            
            constraint_force = shared.joint.node.getObject('dofs').force[0]
            constraint_ref = [0, 3, 0, 0, 0, 0]
        
            net_force  = shared.body.node.getObject('dofs').force[0]
            net_ref = [0, 1, 0, 0, 0, 0]
        
            constraint_check = (vec(constraint_force) - vec(constraint_ref)).norm()
            net_check = (vec(net_force) - vec(net_ref)).norm()
    
            # up to the collision detection precision
            epsilon = 1e-10
            result = (constraint_check < epsilon) and (net_check < epsilon)
            
            if result:
                self.node.sendScriptEvent('success', 0)
            else:
                print 'constraint check:', constraint_check
                print 'net check:', net_check
                self.node.sendScriptEvent('failure', 0)
            
            
            
    


