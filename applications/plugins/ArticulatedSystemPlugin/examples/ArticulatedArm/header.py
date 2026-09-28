
def addHeader(rootNode):

    rootNode.addObject('RequiredPlugin', name='SofaPlugins', pluginName=[
        'ArticulatedSystemPlugin',
        'SofaPython3',
        'Sofa.Component.AnimationLoop',
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.Mapping.NonLinear',
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.Setting',
        'Sofa.Component.SolidMechanics.Spring',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
    ])
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideForceFields hideWireframe')

    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('BlockGaussSeidelConstraintSolver', maxIterations=50, tolerance=1e-5, printLog=False)
    rootNode.addObject('BackgroundSetting', color=[1., 1., 1., 1.])
    rootNode.findData('dt').value=0.01
    rootNode.gravity = [0,-9810,0]


def createScene(rootNode):

    addHeader(rootNode)
