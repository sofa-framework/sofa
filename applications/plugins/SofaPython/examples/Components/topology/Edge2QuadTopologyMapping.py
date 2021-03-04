import Sofa

def createScene(rootNode):

    rootNode.createObject('VisualStyle', displayFlags='showVisualModels showWireframe')
    rootNode.createObject('RequiredPlugin', pluginName='SofaGeneralEngine SofaTopologyMapping SofaPython SofaOpenglVisual SofaMiscMapping')

    positions = [[0,0,k*10] for k in range(4)]
    for i in range(4):
        mesh = rootNode.createChild('Mesh'+str(i+1))
        edgetopo = mesh.createChild('Edge')
        edgetopo.createObject('EdgeSetTopologyContainer', position=positions, edges=[[k,k+1] for k in range(3)])
        edgetopo.createObject('EdgeSetTopologyModifier')
        edgetopo.createObject('MechanicalObject', template='Rigid3', position=[positions[k]+[0,-0.707,0,0.707] for k in range(4)], translation=[40*i,0,0], showObject=True, showObjectScale=2)

        quadtopo = edgetopo.createChild('Quad')
        quadtopo.createObject('QuadSetTopologyContainer')
        quadtopo.createObject('QuadSetTopologyModifier')
        quadtopo.createObject('MechanicalObject')
        quadtopo.createObject('Edge2QuadTopologicalMapping',
                              input=edgetopo.EdgeSetTopologyContainer.getLinkPath(),
                              output=quadtopo.QuadSetTopologyContainer.getLinkPath(),
                              flipNormals=True, nbPointsOnEachCircle=16, radius=10,
                              radiusFocal= [0,15,15,20][i], focalAxis=[[0,0,1],[0,0,1],[0,1,0],[0,1,1]][i])
        quadtopo.createObject('OglModel', color=[1,1,1,1])

    return
