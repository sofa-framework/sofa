import Sofa
import Plugin

import math

from Compliant import Rigid, Vec, Quaternion, Tools

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

    # friction coefficient
    shared.mu = 0.5

    scene = Tools.scene( node )

    manager = node.getObject('manager')
    manager.response = 'FrictionCompliantContact'
    manager.responseParams = 'mu=' + str(shared.mu)

    ode = node.getObject('ode')
    ode.stabilization = 'false'

    num = node.createObject('SequentialSolver',
                            name = 'num',
                            iterations = 100,
                            precision = 1e-14)
    
    proximity = node.getObject('proximity')

    proximity.alarmDistance = 0.5
    proximity.contactDistance = 0.1

    # ground
    ground = Rigid.Body('ground');
    ground.node = ground.insert( scene );

    # plane
    plane = Rigid.Body('plane')
    plane.visual = dir + '/ground.obj'
    plane.collision = plane.visual
    plane.mass_from_mesh( plane.visual, 10 )
    plane.node = plane.insert( scene )
    
    ground.node.createObject('FixedConstraint', 
                             indices = '0')

    # ground-plane joint
    frame = Rigid.Frame()
    frame.translation = [5, 0, 0]

    joint = Rigid.RevoluteJoint(2)
    joint.absolute(frame, ground.node, plane.node)

    joint.node = joint.insert( scene )

    # joint limit 
    limit = joint.node.createChild("limit")
    limit.createObject('MechanicalObject', template = 'Vec1d', position = '0')
    projection = limit.createObject('ProjectionMapping', template = 'Vec6d, Vec1d' )
    projection.set = '0   0 0 0 0 0 -1'
    limit.createObject('UniformCompliance', template = 'Vec1d', compliance = '0' )
    limit.createObject('UnilateralConstraint');
    
    # box
    box = Rigid.Body('box')
    box.visual = dir + '/cube.obj'
    box.collision = box.visual
    box.dofs.translation = [0, 3, 0]
    box.mass_from_mesh( box.visual, 50 )
    box.node = box.insert( scene )

    # constant force field
    ff = joint.node.createObject('ConstantForceField')
    ff.forces = '0 0 0 0 0 -3e5'

    shared.plane = plane.node.getObject('dofs')
    shared.box = box.node.getObject('dofs')
    shared.joint = joint.node.getObject('dofs')

    shared.first = None
   

# scene controller
class Controller(Sofa.PythonScriptController):
     
    def onLoaded(self,node):
        return 0
          
    def reset(self):
        # shared.first = None
        return 0
          
    def onBeginAnimationStep(self, dt):

        relative = Rigid.Frame( shared.plane.position[0] ).inv() * Rigid.Frame( shared.box.position[0] )
        tangent = [relative.translation[0], relative.translation[2]]

        relative_speed = Vec.diff(shared.box.velocity[0][:3], shared.plane.velocity[0][:3])

        local = Quaternion.rotate(Quaternion.conj( Rigid.Frame( shared.plane.position[0] ).rotation ),
                                  relative_speed)
        
        tangent_speed = [local[0], local[2]]

        print 'plane/box relative speed (norm): ', Vec.norm(tangent_speed)
        
        alpha = math.fabs( shared.joint.position[0][-1] )
        mu = math.tan( alpha )
        print 'plane/ground angle:', alpha , 'mu = ', mu
        
        if mu > shared.mu:
            print '(should move)'
 
        print
            
    
        return 0
          
    def bwdInitGraph(self,node):
        return 0
