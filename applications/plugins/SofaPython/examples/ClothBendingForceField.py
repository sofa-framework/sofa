import Sofa
import sys,os,math
import numpy as np


#Utils functions

  
def createClothTopology(parent,name,min,max,resolution):
    size = np.array([0,0,0])
    size=max-min
    parent.createObject("GridMeshCreator", name='loader', resolution = "%i %i"%(resolution,resolution), rotation="90 0 0 ", 
        scale3d ="%f %f %f"%(size[0],size[2],1), translation ="%f %f %f "%(min[0],min[1],min[2]),  trianglePattern = 0)
    parent.createObject("GridMeshCreator", name='loader2', resolution = "%i %i"%(resolution,resolution), rotation="90 0 0 ", 
        scale3d ="%f %f %f"%(size[0],size[2],1), translation ="%f %f %f "%(min[0],min[1],min[2]),  trianglePattern = 1)
    parent.createObject("MeshTopology", name=name, position="@loader.position", edges="@loader.edges", quads="@loader.quads")
    return


def createClothInternalForces(parent, ks=100, kd=0, cb=1, kb=100):
    parent.createObject("ClothSpringForceField", stiffness=ks, damping=kd)
    parent.createObject("ClothBendingForceField", stiffness=kb, damping=kd, bending = cb)
    return
  

def createScene(root):

     ###########################################
    # Variables
    ###########################################
    resolution = 60
    particleMass = 0.001
    totalMass=resolution*resolution*particleMass
    min = np.array([0,0,0])
    max = np.array([1,0,1])
    fixed = "0 %i"%(resolution-1)
    
    ks = 1000.0
    kd = 0.1
    cb = 0.3
    kb = 100*cb

    
    ###########################################
    # Scene creation
    ###########################################
    
    root.createObject('VisualStyle', displayFlags = "showVisual hideMapping hideBehavior showCollisionModels  hideWireframe " )

    root.gravity = [0, -10, 0]
    root.dt = 0.01
    root.createObject('EulerImplicitSolver', rayleighStiffness = 0., rayleighMass=0.)
    root.createObject('CGLinearSolver', iterations=100, tolerance=1e-8, threshold=1e-8, warmStart=1)

    # cloth
    cloth = root.createChild("cloth")
    createClothTopology(cloth,"mesh", min, max, resolution)
    cloth.createObject("MechanicalObject", name="dof",  src="@mesh", velocity = "0 0 0", rotation2 = "0 0 0", translation = "0 0 0", showObject=0, showIndices=0)
    cloth.createObject("UniformMass", rayleighMass=0, totalmass=totalMass)
    cloth.createObject("FixedConstraint", indices=fixed)
    
    createClothInternalForces(cloth,ks,kd,cb,kb)

    # visual model
    visuNode = cloth.createChild('visu')
    visuNode.createObject('OglModel', template='ExtVec3f', name='visual', color="red")
    visuNode.createObject("IdentityMapping")
    
    return
    
   


