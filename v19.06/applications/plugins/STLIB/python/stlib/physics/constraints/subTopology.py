# -*- coding: utf-8 -*-
import Sofa


def SubTopology(attachedTo=None,
                containerLink=None,
                boxRoiLink=None,
                linkType='tetrahedron',
                name="modelSubTopo",
                poissonRatio=0.3,
                youngModulus=18000):
    """
    Args:

        attachedTo (Sofa.Node): Where the node is created.

        containerLink (str): path to container with the dataField to link, ie: '@../container.position'

        boxRoiLink (str): path to boxRoi with the dataField to link,

        linkType (str):	indicate of which type is the subTopo,
                        ei: 'triangle' -> TriangleSetTopologyContainer & TriangleFEMForceField

        name (str):	name of the child node

        youngModulus (float):  The young modulus.

        poissonRatio (float):  The poisson parameter.

    Structure:
        .. sourcecode:: qml

            Node : {
                name : "modelSubTopo",
                TetrahedronSetTopologyContainer,
                TetrahedronFEMForceField
            }
    """
    if attachedTo is None:
        Sofa.msg_error("Your SubTopology isn't child of any node, please set the argument attachedTo")
        return None

    if containerLink is None or boxRoiLink is None:
        Sofa.msg_error("You have to specify at least a container & boxROI link ")
        return None

    linkTypeUp = linkType[0].upper()+linkType[1:]

    # From the linkType which type of topology & forcefield to create :
    # ex : tetrahedra -> create  TetrahedraSetTopologyContainer & TetrahedraFEMForceField
    # From the containerLink we have the position of our TetrahedraSetTopologyContainer
    # From the boxRoiLink we have the elements we want from our boxROI

    # here we extract from the boxRoiLink the name of the data field to link it to
    strTmp = boxRoiLink[::-1]
    start = strTmp.find('nI')+2
    end = strTmp.find('.')
    strTmp = strTmp[start:end]
    argument = strTmp[::-1]

    modelSubTopo = attachedTo.createChild(name)
    modelSubTopo.createObject(linkTypeUp+'SetTopologyContainer',
                              name='container',
                              position=containerLink)

    modelSubTopo.getObject('container').findData(argument).value = boxRoiLink

    modelSubTopo.createObject(linkTypeUp+'FEMForceField',
                              name='FEM',
                              method='large',
                              poissonRatio=poissonRatio,
                              youngModulus=youngModulus)

    return modelSubTopo


# Exemple scene of the 3 differents subTopology working
# tetrahedron/triangle/hexahedron
def createScene(rootNode):
    from stlib.scene import MainHeader
    from stlib.physics.deformable import ElasticMaterialObject

    MainHeader(rootNode)

    # Tetrahedron and triangle subtopology
    target = ElasticMaterialObject(volumeMeshFileName="mesh/liver2.msh",
                                   totalMass=0.5,
                                   attachedTo=rootNode)

    target.createObject('BoxROI', name='boxROI', box=[-20, -20, -20, 20, 20, 20], drawBoxes=True)

    SubTopology(attachedTo=target,
                containerLink='@../container.position',
                boxRoiLink='@../boxROI.tetrahedraInROI',
                name='Default-tetrahedron')

    SubTopology(attachedTo=target,
                containerLink='@../container.position',
                linkType='triangle',
                boxRoiLink='@../boxROI.trianglesInROI',
                name='Triangles')

    # Hexahedron subtopology : (eulalie) Need fix, HexahedronFEMForceField segfault...
    # target = ElasticMaterialObject(volumeMeshFileName="mesh/SimpleBeamHexa.msh",
    #                                totalMass=0.5,
    #                                attachedTo=rootNode)
    # target.removeObject(target.container)
    # target.removeObject(target.forcefield)
    # target.createObject("HexahedronSetTopologyContainer", name="container", position=target.loader.position, hexahedra=target.loader.hexahedra)
    # target.createObject("HexahedronFEMForceField", name="forcefield")

    # target.createObject('BoxROI', name='boxROI', box=[-20, -20, -20, 20, 20, 20], drawBoxes=True)

    # SubTopology(attachedTo=target,
    #             containerLink='@../container.position',
    #             linkType='hexahedron',
    #             boxRoiLink='@../boxROI.hexahedraInROI',
    #             name='Hexahedron')
