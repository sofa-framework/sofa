
import Sofa
import sys
from SofaPython import Tools





def createScene(root):
    
    root.dt = 0.01
    root.gravity = [0, -10, 0]
    
        
    root.createObject('VisualStyle', name="visualStyle1",  displayFlags="hideVisual showBehaviorModels showForceFields showInteractionForceFields showMechanicalMappings" )

    
    root.createObject('RequiredPlugin', pluginName="Compliant")
    #root.createObject('CompliantAttachButton')
    
    
    stiffnessNode = root.createChild("stiffness")
    
    
    
    # A SIMPLE SPRING BETWEEN TWO PARTICLES IN THE SAME MECHANICALOBJECT
    simpleNode = stiffnessNode.createChild("SimpleForceField")
    simpleNode.createObject('CompliantImplicitSolver',debug="0" )
    simpleNode.createObject('MinresSolver' )
    #simpleNode.createObject('EulerImplicitSolver')
    #simpleNode.createObject('CGLinearSolver' )
    simpleNode.createObject('MeshTopology', name="mesh", position="0 2 0  1 2 0", edges="0 1" )
    simpleNode.createObject('MechanicalObject', template="Vec3d", name="defoDOF", showObject="1", showObjectScale="0.05", drawMode=1 )
    simpleNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    simpleNode.createObject('UniformMass',  name="mass", mass="1")
    simpleNode.createObject('StiffSpringForceField',  name="ff", spring="0 1 1e4 0 1")
    
    
    # A INTERACTION SPRING BETWEEN TWO PARTICLES IN SEPARATED MECHANICALOBJECTS
    multiNode = stiffnessNode.createChild("InteractionForceField")
    multiNode.createObject('CompliantImplicitSolver',debug="0" )
    multiNode.createObject('MinresSolver' )
    #multiNode.createObject('EulerImplicitSolver')
    #multiNode.createObject('CGLinearSolver' )
    
    firstNode = multiNode.createChild("0")
    firstNode.createObject('MechanicalObject', template="Vec3d", name="DOF0", showObject="1", showObjectScale="0.05", drawMode=1, position="0 1 0" )
    firstNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    firstNode.createObject('UniformMass',  name="mass", mass="1")
    
    secondNode = multiNode.createChild("1")
    secondNode.createObject('MechanicalObject', template="Vec3d", name="DOF1", showObject="1", showObjectScale="0.05", drawMode=1, position="1 1 0" )
    secondNode.createObject('UniformMass',  name="mass", mass="1")
    
    multiNode.createObject('StiffSpringForceField', name="iff", object1="@0/DOF0", object2="@1/DOF1", spring="0 0 1e4 0 1" )
    
    
    
    
    
   
    
    # A SIMPLE DISTANCEMAPPING BETWEEN TWO PARTICLES IN THE SAME MECHANICALOBJECT
    simpleNode = stiffnessNode.createChild("DistanceMapping")
    simpleNode.createObject('CompliantImplicitSolver',debug="0" )
    simpleNode.createObject('MinresSolver' )
    #simpleNode.createObject('EulerImplicitSolver')
    #simpleNode.createObject('CGLinearSolver' )
    simpleNode.createObject('MeshTopology', name="mesh", position="0 0 0  1 0 0", edges="0 1" )
    simpleNode.createObject('MechanicalObject', template="Vec3d", name="defoDOF", showObject="1", showObjectScale="0.05", drawMode=1 )
    simpleNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    simpleNode.createObject('UniformMass',  name="mass", mass="1")
    behaviorNode = simpleNode.createChild("behavior")
    behaviorNode.createObject('MechanicalObject', template="Vec1d",  name="extensionsDOF" )
    behaviorNode.createObject('EdgeSetTopologyContainer', edges="@../mesh.edges" )
    behaviorNode.createObject('DistanceMapping', showObjectScale="0.02"  )
    behaviorNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-4,  isCompliance=0 )
    



    # A TWO-LAYERS DISTANCEMAPPING BETWEEN TWO PARTICLES IN SEPARATED MECHANICALOBJECTS
    multiNode = stiffnessNode.createChild("SubsetMultiMapping+DistanceMapping")
    multiNode.createObject('CompliantImplicitSolver',debug="0" )
    multiNode.createObject('MinresSolver' )
    #multiNode.createObject('EulerImplicitSolver')
    #multiNode.createObject('CGLinearSolver' )
    
    firstNode = multiNode.createChild("0")
    firstNode.createObject('MechanicalObject', template="Vec3d", name="DOF0", showObject="1", showObjectScale="0.05", drawMode=1, position="0 -1 0" )
    firstNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    firstNode.createObject('UniformMass',  name="mass", mass="1")
    
    secondNode = multiNode.createChild("1")
    secondNode.createObject('MechanicalObject', template="Vec3d", name="DOF1", showObject="1", showObjectScale="0.05", drawMode=1, position="1 -1 0" )
    secondNode.createObject('UniformMass',  name="mass", mass="1")
    
    allNode = firstNode.createChild("all")
    allNode.createObject('MechanicalObject', template="Vec3d", name="allDOF" )
    allNode.createObject('SubsetMultiMapping',  template="Vec3d,Vec3d", input='@../DOF0 @../../1/DOF1', output='@allDOF', indexPairs="0 0  1 0"  )
    secondNode.addChild( allNode )
    
    behaviorNode = allNode.createChild("behavior")
    behaviorNode.createObject('MechanicalObject', template="Vec1d",  name="extensionsDOF" )
    behaviorNode.createObject('EdgeSetTopologyContainer', edges="0 1" )
    behaviorNode.createObject('DistanceMapping', showObjectScale="0.02"  )
    behaviorNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-4,  isCompliance=0 )
    
    
    
    # A DISTANCEMULTIMAPPING BETWEEN TWO PARTICLES IN SEPARATED MECHANICALOBJECTS
    multiNode = stiffnessNode.createChild("DistanceMultiMapping")
    multiNode.createObject('CompliantImplicitSolver',debug="0" )
    multiNode.createObject('MinresSolver' )
    #multiNode.createObject('EulerImplicitSolver')
    #multiNode.createObject('CGLinearSolver' )
    
    firstNode = multiNode.createChild("0")
    firstNode.createObject('MechanicalObject', template="Vec3d", name="DOF0", showObject="1", showObjectScale="0.05", drawMode=1, position="0 -2 0" )
    firstNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    firstNode.createObject('UniformMass',  name="mass", mass="1")
    
    secondNode = multiNode.createChild("1")
    secondNode.createObject('MechanicalObject', template="Vec3d", name="DOF1", showObject="1", showObjectScale="0.05", drawMode=1, position="1 -2 0" )
    secondNode.createObject('UniformMass',  name="mass", mass="1")
    
    behaviorNode = firstNode.createChild("behavior")
    behaviorNode.createObject('MechanicalObject', template="Vec1d", name="extensionsDOF" )
    behaviorNode.createObject('EdgeSetTopologyContainer', edges="0 1" )
    behaviorNode.createObject('DistanceMultiMapping',  template="Vec3d,Vec1d", input='@../DOF0 @../../1/DOF1', output='@extensionsDOF', indexPairs="0 0  1 0", showObjectScale="0.02"  )
    behaviorNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-4,  isCompliance=0 )
    secondNode.addChild( behaviorNode )
    
    
    
    
    
    
    
    
    constraintNode = root.createChild("constraint")
    
    
    
    # A SIMPLE SPRING BETWEEN TWO PARTICLES IN THE SAME MECHANICALOBJECT
    #simpleNode = constraintNode.createChild("SimpleForceField")
    #simpleNode.createObject('CompliantImplicitSolver',debug="0", neglecting_compliance_forces_in_geometric_stiffness=False )
    #simpleNode.createObject('MinresSolver' )
    ##simpleNode.createObject('EulerImplicitSolver')
    ##simpleNode.createObject('CGLinearSolver' )
    #simpleNode.createObject('MeshTopology', name="mesh", position="2 2 0  3 2 0", edges="0 1" )
    #simpleNode.createObject('MechanicalObject', template="Vec3d", name="defoDOF", showObject="1", showObjectScale="0.05", drawMode=1 )
    #simpleNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    #simpleNode.createObject('UniformMass',  name="mass", mass="1")
    #behaviorNode = simpleNode.createChild("behavior")
    #behaviorNode.createObject('MechanicalObject', template="Vec3d",  name="constraintDOF" )
    #behaviorNode.createObject('StiffSpringForceField',  name="ff", spring="0 1 1e4 0 1", isCompliance="1")   # getC is not implemented
    #behaviorNode.createObject('ConstraintValue')
    #behaviorNode.createObject('IdentityMapping')
    
    # A INTERACTION SPRING BETWEEN TWO PARTICLES IN SEPARATED MECHANICALOBJECTS
    # NOT POSSIBLE
   
    
   
    
    ## A SIMPLE DISTANCEMAPPING BETWEEN TWO PARTICLES IN THE SAME MECHANICALOBJECT
    simpleNode = constraintNode.createChild("DistanceMapping")
    simpleNode.createObject('CompliantImplicitSolver',debug="0", neglecting_compliance_forces_in_geometric_stiffness=False )
    simpleNode.createObject('MinresSolver' )
    #simpleNode.createObject('EulerImplicitSolver')
    #simpleNode.createObject('CGLinearSolver' )
    simpleNode.createObject('MeshTopology', name="mesh", position="2 0 0  3 0 0", edges="0 1" )
    simpleNode.createObject('MechanicalObject', template="Vec3d", name="defoDOF", showObject="1", showObjectScale="0.05", drawMode=1 )
    simpleNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    simpleNode.createObject('UniformMass',  name="mass", mass="1")
    behaviorNode = simpleNode.createChild("behavior")
    behaviorNode.createObject('MechanicalObject', template="Vec1d",  name="extensionsDOF" )
    behaviorNode.createObject('EdgeSetTopologyContainer', edges="@../mesh.edges" )
    behaviorNode.createObject('DistanceMapping', showObjectScale="0.02"  )
    behaviorNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-4,  isCompliance=1 )
    behaviorNode.createObject('ConstraintValue')
    



    # A TWO-LAYERS DISTANCEMAPPING BETWEEN TWO PARTICLES IN SEPARATED MECHANICALOBJECTS
    multiNode = constraintNode.createChild("SubsetMultiMapping+DistanceMapping")
    multiNode.createObject('CompliantImplicitSolver',debug="0", neglecting_compliance_forces_in_geometric_stiffness=False )
    multiNode.createObject('MinresSolver' )
    #multiNode.createObject('EulerImplicitSolver')
    #multiNode.createObject('CGLinearSolver' )
    
    firstNode = multiNode.createChild("0")
    firstNode.createObject('MechanicalObject', template="Vec3d", name="DOF0", showObject="1", showObjectScale="0.05", drawMode=1, position="2 -1 0" )
    firstNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    firstNode.createObject('UniformMass',  name="mass", mass="1")
    
    secondNode = multiNode.createChild("1")
    secondNode.createObject('MechanicalObject', template="Vec3d", name="DOF1", showObject="1", showObjectScale="0.05", drawMode=1, position="3 -1 0" )
    secondNode.createObject('UniformMass',  name="mass", mass="1")
    
    allNode = firstNode.createChild("all")
    allNode.createObject('MechanicalObject', template="Vec3d", name="allDOF" )
    allNode.createObject('SubsetMultiMapping',  template="Vec3d,Vec3d", input='@../DOF0 @../../1/DOF1', output='@allDOF', indexPairs="0 0  1 0"  )
    secondNode.addChild( allNode )
    
    behaviorNode = allNode.createChild("behavior")
    behaviorNode.createObject('MechanicalObject', template="Vec1d",  name="extensionsDOF" )
    behaviorNode.createObject('EdgeSetTopologyContainer', edges="0 1" )
    behaviorNode.createObject('DistanceMapping', showObjectScale="0.02"  )
    behaviorNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-4,  isCompliance=1 )
    behaviorNode.createObject('ConstraintValue')
    
    
    
    # A DISTANCEMULTIMAPPING BETWEEN TWO PARTICLES IN SEPARATED MECHANICALOBJECTS
    multiNode = constraintNode.createChild("DistanceMultiMapping")
    multiNode.createObject('CompliantImplicitSolver',debug="0", neglecting_compliance_forces_in_geometric_stiffness=False )
    multiNode.createObject('MinresSolver' )
    #multiNode.createObject('EulerImplicitSolver')
    #multiNode.createObject('CGLinearSolver' )
    
    firstNode = multiNode.createChild("0")
    firstNode.createObject('MechanicalObject', template="Vec3d", name="DOF0", showObject="1", showObjectScale="0.05", drawMode=1, position="2 -2 0" )
    firstNode.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    firstNode.createObject('UniformMass',  name="mass", mass="1")
    
    secondNode = multiNode.createChild("1")
    secondNode.createObject('MechanicalObject', template="Vec3d", name="DOF1", showObject="1", showObjectScale="0.05", drawMode=1, position="3 -2 0" )
    secondNode.createObject('UniformMass',  name="mass", mass="1")
    
    behaviorNode = firstNode.createChild("behavior")
    behaviorNode.createObject('MechanicalObject', template="Vec1d", name="extensionsDOF" )
    behaviorNode.createObject('EdgeSetTopologyContainer', edges="0 1" )
    behaviorNode.createObject('DistanceMultiMapping',  template="Vec3d,Vec1d", input='@../DOF0 @../../1/DOF1', output='@extensionsDOF', indexPairs="0 0  1 0", showObjectScale="0.02"  )
    behaviorNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-4,  isCompliance=1 )
    behaviorNode.createObject('ConstraintValue')
    secondNode.addChild( behaviorNode )
    