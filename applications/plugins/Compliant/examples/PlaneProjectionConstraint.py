import Sofa
    
def createScene(root):
  
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehavior showMechanicalMappings" )
    root.dt = 0.01
    root.gravity = [0, 0, 0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    #root.createObject('CompliantAttachButtonSetting')
    
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=0, neglecting_compliance_forces_in_geometric_stiffness=0)
    root.createObject('LDLTSolver', schur=0)
    
    
    root.createObject('MechanicalObject', template = 'Vec3d', name = 'ind_dofs', position = '0 0 0     1 0 0', showObject=True, showObjectScale=0.05, drawMode=1 )
    root.createObject('UniformMass', mass="1")
    
    
    projectionNode = root.createChild( "projected" )
    projectionNode.createObject('MechanicalObject', template = 'Vec3d', name = 'proj_dofs', showObject=True, showObjectScale=0.03, drawMode=2 )
    projectionNode.createObject('ProjectionToTargetPlaneMapping', indices="0  1", origins="1 1 1  1 0 0", normals="-.3 -.4 1  .7 .8 -.2" )
        
    
    constraintNode = projectionNode.createChild( "constraint" )
    constraintNode.createObject('MechanicalObject', template = 'Vec1d', name = 'distance_dofs', position = '0'  )
    constraintNode.createObject('EdgeSetTopologyContainer', edges="0 1  2 3" )
    constraintNode.createObject('DistanceMultiMapping', template="Vec3d,Vec1d", name='mapping', input = '@../../ind_dofs @../proj_dofs', output = '@distance_dofs', indexPairs="0 0  1 0   0 1  1 1", showObjectScale="0.01"   )
    constraintNode.createObject('UniformCompliance', name="ucomp" ,template="Vec1d", compliance=1e-10 )
    root.addChild( constraintNode )
    
    