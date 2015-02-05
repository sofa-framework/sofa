import Sofa

class RigidMassInfo:
    def __init__(self):
        self.mass=0
        self.com=[0,0,0]
        self.diagonal_inertia=[0,0,0]
        self.inertia_rotation=[0,0,0,1]

    def setFromMesh(self, filepath, density = 1, scale3d=[1,1,1]):
        rigidInfo = Sofa.generateRigid( filepath, density, scale3d[0], scale3d[1], scale3d[2] )
        self.mass = rigidInfo[0]
        self.com = rigidInfo[1:4]
        self.diagonal_inertia = rigidInfo[4:7]
        self.inertia_rotation = rigidInfo[7:11]
