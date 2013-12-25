# simple, 1D PID control on 6D joints.
#
# author: maxime.tournier@inria.fr
#
#

import Vec, Rigid, Tools

# 1-dimensional PID, for a Rigid joint
class PID:
    
    def __init__(self, dofs):
        self.dofs = dofs

        # gains
        self.kp = -1
        self.kd = -0
        self.ki = -0
          
        self.ref_pos = 0
        self.ref_vel = 0
        
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
        p = Vec.dot(self.basis, self.dofs.position[0]) - self.ref_pos
        d = Vec.dot(self.basis, self.dofs.velocity[0]) - self.ref_vel
        i = self.integral + dt * p

        return p, i, d

    # you probably want to call this during onBeginAnimationStep from
    # a PythonController (see SofaPython doc)
    def update(self, dt):
        e, e_sum, e_dot = self.pid(dt)
        
        tau = self.kp * e + self.ki * e_sum + self.kd * e_dot
    
        self.integral = e_sum
        self.apply( tau )


# TODO some other controllers ?
