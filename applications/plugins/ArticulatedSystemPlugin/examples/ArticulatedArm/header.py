
def addHeader(rootNode):

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideForceFields hideWireframe')
    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName=['ArticulatedSystemPlugin', 'SofaPython3'])

    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('ProjectedGaussSeidelConstraintSolver', maxIterations=50, tolerance=1e-5, printLog=False)
    rootNode.addObject('BackgroundSetting', color=[1., 1., 1., 1.])
    rootNode.findData('dt').value=0.01
    rootNode.gravity = [0,-9810,0]


def createScene(rootNode):

    addHeader(rootNode)
