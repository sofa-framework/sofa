# simple, 1D PID control on 6D joints.
#
# author: maxime.tournier@inria.fr
#
#

import Vec, Rigid, Tools

from Vec import Proxy as vec

# 1-dimensional PID, for a Rigid joint. *EXPLICIT* pid
class PID:
    
    def __init__(self, dofs, **args):
        # self.dofs = dofs

        # gains
        self.kp = -1
        self.kd = -0
        self.ki = -0
          
        self.pos = 0
        self.vel = 0
        
        # actuation basis
        self.basis = [0, 0, 0, 0, 0, 0]
        
        self.name = 'pid'

        # overrides stuff
        for k in args:
            setattr(self, k, args[k])

        # insert starts here
        node = dofs.getContext().createChild( self.name )

        self.dofs = node.createObject('MechanicalObject', 
                                      name = 'dofs',
                                      template = 'Vec1d',
                                      position = '0')

        node.createObject('ConstantForceField',
                          template = 'Vec1d',
                          forces = '0')
        
        self.map = node.createObject('ProjectionMapping',
                                     set = '0 ' + Tools.cat(self.basis) )
        self.node = node

        self.reset()
    

    def reset(self):
        self.integral = 0

    # applies a 1D torque to the joint, through the wrench basis
    def apply(self, tau): 

        current = self.dofs.externalForce

        if type(current) == list:
            if len(current) == 0:
                value = 0
            else:
                value = current[0][0]
        else:
            value = current

        print value, tau
        value += tau

        self.dofs.externalForce = str(value)

    def pid(self, dt):
        p = self.dofs.position[0][0] - self.pos
        d = self.dofs.velocity[0][0] - self.vel
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
        
        # update mapping just in case
        self.map.set = Tools.cat([0] + self.basis)
        self.map.offset = str(self.pos)
        self.map.init()

        self.update(dt)

    def post_step(self, dt):
        pass

# TODO some other controllers ?



# this is for large gains + zero target velocity
class ImplicitPID:
    
    def __init__(self, dofs, **args):
        # gains
        self.kp = -1
        self.kd = -0
        self.ki = -0
          
        self.pos = 0
        
        # actuation basis
        self.basis = [0, 0, 0, 0, 0, 0]
        
        self.reset()
        
        self.name = 'pid'

        # overrides stuff
        for k in args:
            setattr(self, k, args[k])

        # insert start here
        node = dofs.getContext().createChild( self.name )

        self.dofs = node.createObject('MechanicalObject', 
                                      name = 'dofs',
                                      template = 'Vec1d',
                                      position = '0')
        
        self.map = node.createObject('ProjectionMapping',
                                     set = '0 ' + Tools.cat(self.basis) )
        
        self.ff = node.createObject('UniformCompliance',
                                    template = 'Vec1d',
                                    compliance = '0' )
        
        self.node = node
        

    def reset(self):
        self.integral = 0

    # you need to call this during onBeginAnimationStep from
    # a PythonController (see SofaPython doc)
    def pre_step(self, dt):

        # update mapping just in case
        self.map.set = Tools.cat([0] + self.basis)
        self.map.offset = str(self.pos)
        self.map.init()

        stiff = - self.kp - dt * self.ki
        damping = - self.kd
        
        self.ff.compliance = 1.0 / stiff
        self.ff.damping = damping

        # trigger compliance matrix recomputation
        self.ff.init()

        # net explicit force
        self.explicit = 0
        
        # explicit integral part
        self.apply( self.ki * self.integral )

    # apply an explicit force
    def apply(self, f):
        self.explicit += f
        self.dofs.externalForce = str( self.explicit )


    # force applied at the end of time step
    def post_force(self):
        return self.dofs.force - self.kd * self.dofs.velocity[0][0] + self.explicit

    # call this during onEndAnimationStep
    def post_step(self, dt):

        # update integral with error on time step start
        self.integral = self.integral + dt * self.dofs.position[0][0]
        
        # sanity check
        # check = self.kp * self.dofs.position + self.kd * self.dofs.velocity + self.ki * self.integral
        
        # force = self.post_force()

        # print 'post-step force:', force
        # print 'should be:', check

        


