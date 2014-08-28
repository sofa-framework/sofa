import Sofa

def createScene(root):
    root.createObject('RequiredPlugin', name='Flexible')
    root.createObject('RequiredPlugin', name='image')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
   
    
    root.dt = 0.05
    root.gravity = [0, -9.8, 0]
    
    root.createObject('ImplicitEulerSolver')
    root.createObject('MinResLinearSolver', iteration="100", tolerance="1e-15")
    
    root.createObject('MeshObjLoader', name="loader", filename="mesh/torus.obj", triangulate="1")
    #root.createObject('OglModel', template="ExtVec3f", name="Visual", fileMesh="mesh/torus.obj", color="1 0.8 0.8 ")
    
    root.createObject('MeshToImageEngine', template="ImageUC", name="rasterizer", position="@loader.position", edges="@loader.edges", triangles="@loader.triangles", voxelSize="0.1", padSize="1", rotateImage="true" )
    root.createObject('ImageContainer', template="ImageUC", name="img", src="@rasterizer", drawBB="false" )

    #root.createObject('ImageViewer', template="ImageUC", name="viewer", image="@img.image", transform="@img.transform")
    
    root.createObject('ImageSampler', template="ImageUC", name="sampler", src="@img", method="1", param="2", fixedPosition="2 0 0 -2 0 0" ,printLog="false" )
    root.createObject('MergeMeshes', name="merged", nbMeshes="2", position1="@sampler.fixedPosition", position2="@sampler.position" )

    root.createObject('VoronoiShapeFunction', name="SF", position="@merged.position", src="@img", method="0", nbRef="4" )
        
    
    rigidNode = root.createChild('Rigid')
    rigidNode.createObject( 'MechanicalObject', template="Rigid", name="dof", showObject="true", showObjectScale="0.7", position="@../merged.position1" )
    rigidNode.createObject( 'BoxROI', template="Vec3d", box="0 -2 0 5 2 5", position="@../merged.position1", name="FixedROI")
    rigidNode.createObject( 'FixedConstraint', indices="@FixedROI.indices" )
      
        
    affineNode = root.createChild('Affine')
    affineNode.createObject('MechanicalObject', template="Affine", name="dof", showObject="true", showObjectScale="1.5", position="@../merged.position2" )
    affineNode.createObject('BoxROI', template="Vec3d", box="0 -2 0 5 2 5", position="@../merged.position2", name="FixedROI")
    affineNode.createObject('FixedConstraint', indices="@FixedROI.indices" )
    
    
    behaviorNode = rigidNode.createChild('behavior')
    
    behaviorNode.createObject('ImageGaussPointSampler', name="sampler", indices="@../../SF.indices", weights="@../../SF.weights", transform="@../../SF.transform", method="2", order="1", showSamplesScale="0", printLog="true", targetNumber="200" )
    behaviorNode.createObject('MechanicalObject', template="F331", name="F",  useMask="0",  showObject="1", showObjectScale="0.05" )
    behaviorNode.createObject('LinearMultiMapping', template="Rigid,Affine,F331", input1="@..", input2="@../../Affine", output="@.", printLog="0", showDeformationGradientScale="0", assemble="0")
    
    affineNode.addChild( behaviorNode )

    strainNode = behaviorNode.createChild('E')
    strainNode.createObject('MechanicalObject',  template="E331", name="E"  ) 
    strainNode.createObject('GreenStrainMapping', template="F331,E331"    )
    strainNode.createObject('HookeForceField',  template="E331", name="ff", youngModulus="2000.0", poissonRatio="0.2", viscosity="0") 
                
    massNode = rigidNode.createChild('mass')
    massNode.createObject('TransferFunction',name="densityTF", template="ImageUC,ImageD", inputImage="@../../rasterizer.image", param="0 0 1 0.005")
    massNode.createObject('MechanicalObject', position="@../../merged.position", useMask="0")
    massNode.createObject('ImageDensityMass', template="Vec3d", densityImage="@densityTF.outputImage", transform="@../../rasterizer.transform", lumping="0",  printMassMatrix="true" )
    #massNode.createObject('UniformMass', totalMass="20" )
    massNode.createObject('LinearMultiMapping', template="Rigid,Affine,Vec3d", input1="@..", input2="@../../Affine", output="@.", printLog="0", assemble="0")
    affineNode.addChild( massNode )
     
    visualNode = rigidNode.createChild('visual')
    visualNode.createObject('OglModel', template="ExtVec3f", name="Visual", fileMesh="mesh/torus.obj", color="1 0.8 0.8 ")
    visualNode.createObject('LinearMultiMapping', template="Rigid,Affine,ExtVec3f", input1="@..", input2="@../../Affine", output="@.", printLog="0", assemble="0")
    affineNode.addChild( visualNode )
