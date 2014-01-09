# simple, 1D PID control on 6D joints.
#
# author: maxime.tournier@inria.fr
#
#

import Vec, Rigid, Tools

from Vec import Proxy as vec

# 1-dimensional PID, for a Rigid joint. *EXPLICIT* pid
class PID:
    
    def __init__(self, dofs):
        self.dofs = dofs

        # gains
        self.kp = -1
        self.kd = -0
        self.ki = -0
          
        self.pos = 0
        self.vel = 0
        
        # actuation basis
        self.basis = [0, 0, 0, 0, 0, 0]
        
        self.reset()

    def reset(self):
        self.integral = 0

    # applies a 1D torque to the joint, through the wrench basis
    def apply(self, tau): 

        current = self.dofs.externalForce
        value = [tau * ei for ei in self.basis]
        
        # TODO optimize ? setting the list directly does not work
        # across time steps :-/
        if len(current) == 0:
            self.dofs.externalForce = Tools.cat(value)
        else:
            self.dofs.externalForce = Tools.cat( Vec.sum(current[0], value) )
        
    def pid(self, dt):
        p = Vec.dot(self.basis, self.dofs.position[0]) - self.pos
        d = Vec.dot(self.basis, self.dofs.velocity[0]) - self.vel
        i = self.integral + dt * p

        return p, i, d

    # you probably want to call this during onBeginAnimationStep from
    # a PythonController (see SofaPython doc)
    def update(self, dt):
        e, e_sum, e_dot = self.pid(dt)
        
        tau = self.kp * e + self.ki * e_sum + self.kd * e_dot
    
        self.integral = e_sum
        self.apply( tau )

    # hop
    def pre_step(self, dt):
        self.update(dt)

    def post_step(self, dt):
        pass

# TODO some other controllers ?



# this is for large gains + zero target velocity
class ImplicitPID:
    
    def __init__(self, dofs):
        self.dofs = dofs

        # gains
        self.kp = -1
        self.kd = -0
        self.ki = -0
          
        self.pos = 0
        
        # actuation basis
        # self.basis = vec( [0, 0, 0, 0, 0, 0] )
        
        self.reset()
        
        self.name = 'pid'


    def insert( self ):
        node = self.dofs.getContext().createChild( self.name )

        self.dofs = node.createObject('MechanicalObject', 
                                      template = 'Vec1d',
                                      position = '0')
        
        self.map = node.createObject('ProjectionMapping',
                                     set = '0 ' + Tools.cat(self.basis) )
        
        self.ff = node.createObject('UniformCompliance',
                                    template = 'Vec1d',
                                    compliance = '1e-4' )
        
        self.node = node

    def reset(self):
        self.integral = 0

    # you need to call this during onBeginAnimationStep from
    # a PythonController (see SofaPython doc)
    def pre_step(self, dt):

        self.map.offset = str(self.pos)

        stiff = - self.kp - dt * self.ki
        damping = - self.kd / dt
        
        self.ff.compliance = 1.0 / stiff
        # self.ff.rayleighStiffness = damping/stiff

        # trigger compliance matrix recomputation
        # self.ff.init()

        # integral part
        # self.dofs.externalForce = str( self.ki * self.integral )
        
    # call this during onEndAnimationStep
    def post_step(self, dt):
        # update integral with error on time step start
        self.integral = self.integral - dt * self.dofs.position
        
