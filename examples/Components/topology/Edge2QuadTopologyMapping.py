import Sofa

def createScene(rootNode):

    rootNode.createObject('VisualStyle', displayFlags='showVisualModels showWireframe')
    rootNode.createObject('RequiredPlugin', pluginName='SofaGeneralEngine SofaTopologyMapping SofaPython SofaOpenglVisual SofaMiscMapping')

    positions = [[0,0,0],[10,0,0],[20,0,0],[30,0,0]]
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

    return
