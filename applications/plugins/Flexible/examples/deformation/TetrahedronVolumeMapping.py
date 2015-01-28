import Sofa

## using TetrahedronVolumeMapping to add volume constraint on tetrahedral finite elements.
## 2 methods: a constraint per element or a constrain per node
    
    
def createTeta( node, name, perNodeConstraint, torus ):
    
    
    if perNodeConstraint:
        name += "_perNode"
    else:
        name += "_perTetra"
    
    localNode = node.createChild(name)
    
    localNode.createObject('CompliantImplicitSolver', stabilization="no stabilization")
    localNode.createObject('MinresSolver',iterations="100",precision="1e-10")
    
    if torus:
        localNode.createObject('MeshGmshLoader', name="loader", filename="mesh/torus_low_res.msh" )
        localNode.createObject('Mesh', name="mesh", src="@loader" ) 
    else:
        localNode.createObject('Mesh', name="mesh", position="0 0 0  1 0 0  0 1 0  0 0 1", tetrahedra="0 1 2 3" )
        
    
    localNode.createObject('MechanicalObject', template="Vec3d", name="parent", showObject="1", showObjectScale="3")
    
       
    if torus:
        localNode.createObject('BoxROI', template="Vec3d", box="0 -2 0 5 2 5", position="@mesh.position", name="FixedROI")
        localNode.createObject('FixedConstraint', indices="@FixedROI.indices" )
    else:
        localNode.createObject('FixedConstraint', indices="0" )
    
    localNode.createObject('BarycentricShapeFunction' )
    localNode.createObject('UniformMass',totalMass="250" )
 

    behaviorNode = localNode.createChild("behavior")
    behaviorNode.createObject('TopologyGaussPointSampler', name="sampler", inPosition="@../mesh.position", showSamplesScale="0" ,method="0",order="1" )
    behaviorNode.createObject('MechanicalObject',  template="F331", name="F" , showObject="0" ,showObjectScale="0.05")
    behaviorNode.createObject('LinearMapping', template="Vec3d,F331",assemble=1 )

    strainNode = behaviorNode.createChild("strain")
    strainNode.createObject('MechanicalObject', template="E331", name="E" )
    strainNode.createObject('CorotationalStrainMapping', template="F331,E331", method="svd", assemble=1,geometricStiffness=1 )
    strainNode.createObject('HookeForceField',  template="E331" ,name="ff", assemble=1, youngModulus="2000.0", poissonRatio="0.49",viscosity="0",rayleighStiffness="0",isCompliance="1"  )   

    volNode = localNode.createChild("vol")
    volNode.createObject('MechanicalObject',  template="Vec1d", name="Volume"  )
    volNode.createObject('TetrahedronVolumeMapping', template="Vec3d,Vec1d", applyRestPosition=1, volumePerNodes=perNodeConstraint ) 
    volNode.createObject('UniformCompliance', template="Vec1d", compliance=0, isCompliance="1", damping="0")
    volNode.createObject('ConstraintValue')
    
    
    if perNodeConstraint:
        color="green"
    else:
        color="red"
    visuNode = localNode.createChild("visu")    
    visuNode.createObject('VisualModel', position="@../mesh.position", triangles="@../mesh.triangles",color=color )
    visuNode.createObject('IdentityMapping')
    
    return localNode



    
def createScene(root):
    
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehavior showBehaviorModels" )
    root.dt = 0.05
    root.gravity = [0,-1,0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('RequiredPlugin', pluginName = 'Flexible')
    root.createObject('CompliantAttachButtonSetting')
    
    createTeta( root, "single_perTetra", 0, 0 )
    createTeta( root, "single_perNode", 1, 0 )
    
    createTeta( root, "torus_perTetra", 0, 1 )
    createTeta( root, "torus_perNode", 1, 1 )
    