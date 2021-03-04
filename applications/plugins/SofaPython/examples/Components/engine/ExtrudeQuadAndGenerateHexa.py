import Sofa

def createScene(rootNode):

    rootNode.createObject('VisualStyle', displayFlags='showVisualModels showForceFields showBehavior')
    rootNode.createObject('RequiredPlugin', pluginName='SofaBoundaryCondition SofaGeneralEngine SofaTopologyMapping SofaPython SofaOpenglVisual SofaMiscMapping')

    rootNode.gravity = [0,-9810,0]

    positions = [[0,0,0],[15,0,0],[30,0,0],[45,0,0]]
    edgetopo = rootNode.createChild('Edge')
    edgetopo.createObject('EdgeSetTopologyContainer', position=positions, edges=[[k,k+1] for k in range(3)])
    edgetopo.createObject('EdgeSetTopologyModifier')
    edgetopo.createObject('MechanicalObject', template='Rigid3', position=[positions[k]+[0,0,0,1] for k in range(4)], showObject=True, showObjectScale=2)

    quadtopo = edgetopo.createChild('Quad')
    quadtopo.createObject('QuadSetTopologyContainer')
    quadtopo.createObject('QuadSetTopologyModifier')
    quadtopo.createObject('MechanicalObject')
    quadtopo.createObject('Edge2QuadTopologicalMapping',
                          input=edgetopo.EdgeSetTopologyContainer.getLinkPath(),
                          output=quadtopo.QuadSetTopologyContainer.getLinkPath(),
                          flipNormals=True, nbPointsOnEachCircle=8, radius=10)
    quadtopo.createObject('OglModel', color=[1,1,1,1])

    hexatopo = quadtopo.createChild('Hexa')
    hexatopo.createObject('EulerImplicitSolver')
    hexatopo.createObject('CGLinearSolver')
    hexatopo.createObject('ExtrudeQuadsAndGenerateHexas', name='extruder',
                            surfaceVertices=quadtopo.MechanicalObject.findData('position').getLinkPath(),
                            surfaceQuads=quadtopo.QuadSetTopologyContainer.findData('quads').getLinkPath(),
                            thicknessIn=5, thicknessOut=5, numberOfSlices=2, flipNormals=True)

    hexatopo.createObject('HexahedronSetTopologyContainer',
                            points=hexatopo.extruder.findData('extrudedVertices').getLinkPath(),
                            hexahedra=hexatopo.extruder.findData('extrudedHexas').getLinkPath())
    hexatopo.createObject('HexahedronSetTopologyModifier')
    hexatopo.createObject('MechanicalObject')
    hexatopo.createObject('UniformMass', totalMass=0.5)
    hexatopo.createObject('HexahedronFEMForceField', poissonRatio=0.3, youngModulus=650)
    hexatopo.createObject('FixedConstraint', indices=range(24))

    return
