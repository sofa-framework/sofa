import Sofa
import math

from Compliant import Rigid

# global structure for passing data to controller
class ControlData:
     pass

global control

control = ControlData


def createScene(node):
     node.createObject('RequiredPlugin', pluginName = 'Compliant')

     node.animate = 'true'

     node.createObject('VisualStyle', displayFlags='hideBehaviorModels hideCollisionModels hideMappings hideForceFields')
     node.dt = 0.005
     
     node.gravity = '0 -9.81 0'
     
     ode = node.createObject('CompliantImplicitSolver',
                             name='odesolver' )

     ode.stabilization = "pre-stabilization"
     # ode.debug = 'true'

     num = node.createObject('MinresSolver',
                             name = 'numsolver',
                             iterations = '250',
                             precision = '1e-14')
     
     node.createObject('PythonScriptController', 
                       filename = __file__,
                       classname = 'Controller' )
     
     scene = node.createChild('scene') 

     inertia_forces = 'true'

     # dofs
     base = Rigid.Body('base')
     base.node = base.insert( scene )
     base.visual = 'mesh/box.obj'
     base.node.createObject('FixedConstraint', indices = '0' )

     link1 = Rigid.Body('link1')
     link1.dofs.translation = [0, 0, 0]
     link1.visual = 'mesh/cylinder.obj'
     link1.inertia_forces = inertia_forces
     link1.mass_from_mesh( link1.visual )
     link1.node = link1.insert( scene )
     
     link2 = Rigid.Body('link2')
     link2.dofs.translation = [0, 10, 0]
     link2.visual = 'mesh/cylinder.obj'
     link2.mass_from_mesh( link2.visual )
     link2.inertia_forces = inertia_forces
     link2.node = link2.insert( scene )
     
     # joints
     joint1 = Rigid.RevoluteJoint(2)
     joint1.append(base.node, Rigid.Frame().read('0 0 0 0 0 0 1') )
     joint1.append(link1.node, Rigid.Frame().read('0 0 0 0 0 0 1') )
     joint1.node = joint1.insert(scene)

     joint2 = Rigid.RevoluteJoint(2)
     joint2.append(link1.node, Rigid.Frame().read('0 10 0 0 0 0 1') )
     joint2.append(link2.node, Rigid.Frame().read('0 0 0 0 0 0 1') )
     joint2.node = joint2.insert(scene)

     # control
     control.joint1 = ControlledJoint( joint1.node.getObject('dofs') )

     control.joint2 = ControlledJoint( joint2.node.getObject('dofs') )

     # pid controller reference pos for joint2
     control.joint2.ref_pos = [0, 0, 0, 0, 0, math.pi / 2]

     return node



class ControlledJoint:
     def __init__(self, dofs):
          self.dofs = dofs

          scale = 1e7

          self.kp = -1 * scale
          self.kd = -1 * scale
          self.ki = -1 * scale
          
          self.integral = [0, 0, 0, 0, 0, 0]
          self.ref_pos = [0, 0, 0, 0, 0, 0]
          self.ref_vel = [0, 0, 0, 0, 0, 0]
      
     def pid(self):
          
          # error p/d
          e = [ p - r for p, r in zip(self.dofs.position[0], self.ref_pos)]
          e_dot = [ v - r for v, r in zip(self.dofs.velocity[0], self.ref_vel)]

          return [ self.kp * p + self.ki * i + self.kd * d
                   for p, i, d in zip(e, self.integral, e_dot) ], e
     
     def update(self, dt):
          tau, e = self.pid()
          self.dofs.externalForce = Rigid.concat(tau)
          self.integral = [i + dt * f for i, f in zip(self.integral, e) ]
          
          # print 'error: ' + str(e)
          # print 'tau: ' + str(tau)
          


class Controller(Sofa.PythonScriptController):
     
     def onLoaded(self,node):
          return 0
          
     def onBeginAnimationStep(self, dt):

          control.joint1.update(dt)
          control.joint2.update(dt)

          return 0

     def bwdInitGraph(self,node):
          return 0


