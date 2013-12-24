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

    def apply(self, tau): 
        self.dofs.externalForce = Tools.cat( [tau * ei for ei in self.basis] )

    def pid(self):
        p = Vec.dot(self.basis, self.dofs.position[0]) - self.ref_pos
        d = Vec.dot(self.basis, self.dofs.velocity[0]) - self.ref_vel
        i = self.integral + dt * p

        return p, i, d

    # you probably want to call this during onBeginAnimationStep
    def update(self, dt):
        e, e_sum, e_dot = self.pid()
        
        tau = self.kp * e + self.ki * e_sum + self.kd * e_dot
    
        self.integral = e_sum
        self.apply( tau )
