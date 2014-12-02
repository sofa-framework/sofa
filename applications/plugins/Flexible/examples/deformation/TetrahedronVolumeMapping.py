
import Sofa

from Compliant import StructuralAPI, Tools
import sys

path = Tools.path( __file__ )+"/"


    
                
    
def createScene(root):
    
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehavior showWireframe showBehaviorModels" )
    root.dt = 0.05
    root.gravity = [0,-1,0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('RequiredPlugin', pluginName = 'Flexible')
    root.createObject('CompliantAttachButtonSetting', isCompliance=0, compliance=1e-8)
    
           
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization="no stabilization")
    root.createObject('LDLTSolver')
    
    root.createObject('Mesh', name="mesh", position="0 0 0  1 0 0  0 1 0  0 0 1", tetrahedra="0 1 2 3" )

    root.createObject('MechanicalObject', template="Vec3d", name="parent", showObject="1", showObjectScale="10")
    root.createObject('FixedConstraint', indices="0" )
    root.createObject('BarycentricShapeFunction' )
    root.createObject('UniformMass' )

    behaviorNode = root.createChild("behavior")
    behaviorNode.createObject('TopologyGaussPointSampler', name="sampler", inPosition="@../mesh.position", showSamplesScale="0" ,method="0",order="1" )
    behaviorNode.createObject('MechanicalObject',  template="F331", name="F" , showObject="0" ,showObjectScale="0.05")
    behaviorNode.createObject('LinearMapping', template="Vec3d,F331",assemble=1 )

    strainNode = behaviorNode.createChild("strain")
    strainNode.createObject('MechanicalObject', template="E331", name="E" )
    strainNode.createObject('CorotationalStrainMapping', template="F331,E331", method="svd", assemble=1 )
    strainNode.createObject('HookeForceField',  template="E331" ,name="ff", assemble=1, youngModulus="2000.0", poissonRatio="0"  )   

    volNode = root.createChild("vol")
    volNode.createObject('MechanicalObject',  template="Vec1d", name="Volume"  )
    volNode.createObject('TetrahedronVolumeMapping', template="Vec3d,Vec1d", applyRestPosition=1 ) 
    volNode.createObject('UniformCompliance', template="Vec1d", compliance="0", isCompliance="1", damping="0")
    volNode.createObject('ConstraintValue')
          