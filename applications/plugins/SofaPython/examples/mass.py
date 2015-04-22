import SofaPython.mass

m = SofaPython.mass.RigidMassInfo()
m.setFromMesh("data/dragon.obj")
print "mass:", m.mass, "- com:", m.com, "- diagonal_inertia:", m.diagonal_inertia, "- inertia_rotation:", m.inertia_rotation

def createScene(rootNode):
    pass