import os
path=os.path.dirname(os.path.abspath(__file__))+'/data/'
basePath = path+'RobotBase.stl'
part11Path = path+'RobotPart1.stl'
part12Path = path+'RobotPart3.stl'
part21Path = path+'RobotPart2.stl'
part22Path = path+'RobotPart4.stl'
part3Path = path+'RobotPart6.stl'
part41Path = path+'RobotPart5.stl'
part42Path = path+'RobotPart9.stl'
part51Path = path+'RobotPart7.stl'
part52Path = path+'RobotPart8.stl'
part6Path = path+'RobotPart10.stl'


def addVisu(node, index, filename, translation=[0, 0, 0]):

    if filename is None:
        return

    visu = node.addChild('Visu'+str(index))
    visu.addObject('MeshSTLLoader', name='loader', filename=filename, translation=translation)
    visu.addObject('MeshTopology', src='@loader')
    visu.addObject('OglModel', color=[1.0,0.8,0.0,1.])
    visu.addObject('RigidMapping')

    return


def addCenter(node, name,
              parentIndex, childIndex,
              posOnParent, posOnChild,
              articulationProcess,
              isTranslation, isRotation, axis,
              articulationIndex):

    center = node.addChild(name)
    center.addObject('ArticulationCenter', parentIndex=parentIndex, childIndex=childIndex, posOnParent=posOnParent, posOnChild=posOnChild, articulationProcess=articulationProcess)

    articulation = center.addChild('Articulation')
    articulation.addObject('Articulation', translation=isTranslation, rotation=isRotation, rotationAxis=axis, articulationIndex=articulationIndex)

    return center


def addPart(node, name, index, filename1, filename2=None, translation=[0, 0, 0]):

    part = node.addChild(name)
    part.addObject('MechanicalObject', template='Rigid3', position=[0,0,0,0,0,0,1])
    part.addObject('RigidMapping', index=index, globalToLocalCoords=True)

    addVisu(part, 1, filename1, translation=translation)
    addVisu(part, 2, filename2, translation=translation)

    return part


class Robot:

    def __init__(self, node):
        self.node=node

    def addRobot(self, name='Robot', translation=[0,0,0]):

        # Positions of parts
        positions = [
                    [160.8,     0, 160.8, 0,0,0,1],
                    [160.8,  78.5, 160.8, 0,0,0,1],
                    [254.8,   171, 160.8, 0,0,0,1],
                    [347.3,   372, 160.8, 0,0,0,1],
                    [254.8, 569.6, 160.8, 0,0,0,1],
                    [160.8, 500.5, 160.8, 0,0,0,1],
                    [160.8, 442.5, 160.8, 0,0,0,1]
                    ]

        # You can change the joint angles here
        initAngles = [0,0,0,0,0,0]

        # Robot node
        robot = self.node.addChild(name)
        robot.addData('angles', initAngles, None, 'angle of articulations in radian', '', 'vector<float>')
        robot.addObject('EulerImplicitSolver')
        robot.addObject('SparseLDLSolver', template="CompressedRowSparseMatrixMat3x3d")
        robot.addObject('GenericConstraintCorrection')

        # Articulations node
        articulations = robot.addChild('Articulations')
        articulations.addObject('MechanicalObject', name='dofs', template='Vec1', rest_position=robot.getData('angles').getLinkPath(), position=initAngles)
        articulations.addObject('ArticulatedHierarchyContainer')
        articulations.addObject('UniformMass', totalMass=1)
        articulations.addObject('RestShapeSpringsForceField', stiffness=1e10, points=list(range(6)))

        # Rigid
        rigid = articulations.addChild('Rigid')
        rigid.addObject('MechanicalObject', name='dofs', template='Rigid3', showObject=False, showObjectScale=10,
                            position=positions[0:7],
                            translation=translation)
        rigid.addObject('ArticulatedSystemMapping', input1=articulations.dofs.getLinkPath(), output=rigid.dofs.getLinkPath())

        # Visu
        visu = rigid.addChild('Visu')
        addPart(visu, 'Base' , 0, basePath, translation=translation)
        addPart(visu, 'Part1', 1, part11Path, part12Path, translation=translation)
        addPart(visu, 'Part2', 2, part21Path, part22Path, translation=translation)
        addPart(visu, 'Part3', 3, part3Path, translation=translation)
        addPart(visu, 'Part4', 4, part41Path, part42Path, translation=translation)
        addPart(visu, 'Part5', 5, part51Path, part52Path, translation=translation)
        addPart(visu, 'Part6', 6, part6Path, translation=translation)

        # Center of articulations
        centers = articulations.addChild('ArticulationsCenters')
        addCenter(centers, 'CenterBase' , 0, 1, [   0,  78.5, 0], [   0,      0, 0], 0, 0, 1, [0, 1, 0], 0)
        addCenter(centers, 'CenterPart1', 1, 2, [  94,  92.5, 0], [   0,      0, 0], 0, 0, 1, [1, 0, 0], 1)
        addCenter(centers, 'CenterPart2', 2, 3, [92.5,  92.5, 0], [   0, -108.5, 0], 0, 0, 1, [0, 1, 0], 2)
        addCenter(centers, 'CenterPart3', 3, 4, [   0, 108.5, 0], [92.5,  -89.1, 0], 0, 0, 1, [0, 0, 0], 3)
        addCenter(centers, 'CenterPart4', 4, 5, [   0,     0, 0], [  94,   69.1, 0], 0, 0, 1, [1, 0, 0], 4)
        addCenter(centers, 'CenterPart5', 5, 6, [   0,     0, 0], [   0,     58, 0], 0, 0, 1, [0, 1, 0], 5)

        return robot


# Test/example scene
def createScene(rootNode):

    from header import addHeader
    # from robotGUI import RobotGUI  # Uncomment this if you want to use the GUI

    addHeader(rootNode)

    # Robot
    robot = Robot(rootNode).addRobot()
    # robot.addObject(RobotGUI(robot=robot))  # Uncomment this if you want to use the GUI

    return
