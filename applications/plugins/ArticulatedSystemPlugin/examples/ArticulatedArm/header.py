
def addHeader(rootNode):

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideForceFields hideWireframe')
    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName=['ArticulatedSystemPlugin',
     'SofaGeneralAnimationLoop', 'SofaPython3', 'SofaSparseSolver', 'SofaPreconditioner', 'SuctionCup',
     'SofaOpenglVisual', 'SofaBoundaryCondition', 'SofaGeneralRigid', 'SofaConstraint',
     'SofaMiscMapping', 'SofaImplicitOdeSolver', 'SofaLoader', 'SofaSimpleFem', 'SofaTopologyMapping',
     'SofaDeformable', 'SofaEngine', 'SofaRigid', 'SofaGeneralLinearSolver', 'SofaMeshCollision',
     'SofaGeneralEngine'])

    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', maxIterations=50, tolerance=1e-5, printLog=False)
    rootNode.addObject('BackgroundSetting', color=[1., 1., 1., 1.])
    rootNode.findData('dt').value=0.01
    rootNode.gravity = [0,-9810,0]


def createScene(rootNode):

    addHeader(rootNode)
