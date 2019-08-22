import Sofa

import sys

import Compliant
path = Compliant.path()


from Compliant.Vec import Proxy as vec

from Compliant import StructuralAPI, Tools


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
    ode.constraint_forces = "propagate"
    
    # ode.debug = True

    node.gravity = '0 -1 0'
    
    num = node.createObject('SequentialSolver',
                            name = 'num',
                            iterations = 200,
                            precision = 0,
                            iterateOnBilaterals = True)

    resp = node.createObject('DiagonalResponse')

    # NB: either iterateOnBilaterals or use a non-diagonal Response


    manager = node.getObject('manager')
    manager.response = 'CompliantContact'


    style = node.getObject('style')

    style.displayFlags = 'showBehaviorModels showCollisionModels'
    
    proximity = node.getObject('proximity')
    proximity.contactDistance = 0.01
    
    # dofs
    p1 = insert_point(scene, 'p1', [-1, 1, 0]) 
    p2 = insert_point(scene, 'p2', [1, 1, 0]) 
    
    
    rigid = StructuralAPI.RigidBody( scene, 'rigid' )
    rigid.setManually( [0,0,0,0,0,0,1], 1, [1,1,1] )
    rigid.addCollisionMesh( path + '/examples/mesh/ground.obj' )
    
    
    ground = StructuralAPI.RigidBody( scene, 'ground' )
    ground.setManually( [0,0,0,0,0,0,1], 1, [1,1,1] )
    ground.node.createObject('FixedConstraint')
    
    
    
    # blocked joint between ground/rigid
    joint = StructuralAPI.FixedRigidJoint( "joint", ground.node, rigid.node )
    
   
    
    script = node.createObject('PythonScriptController',
                               filename = __file__,
                               classname = 'Controller')
    
    return node


class Controller(Sofa.PythonScriptController):
    
    def onLoaded(self,node):
        self.node = node
        self.joint = node.getObject("scene/rigid/joint/dofs")
        self.body = node.getObject("scene/rigid/dofs")

    def onEndAnimationStep(self, dt):

        # wait for simulation to settle
        if self.node.getRoot().getTime() > 1: 
            
            constraint_force = self.joint.force[0]
            constraint_ref = [0, 3, 0, 0, 0, 0]
        
            net_force  = self.body.force[0]
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
            
            
            
    


