import Sofa

import os

g_size = 10;

def createScene(root_node, template):

    size = int(os.getenv('s', g_size)) # read the value for size from the env, fallback to g_size if the env.var not set

    root_node.dt = 0.04

    root_node.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Projective')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.Mass')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Grid')
    root_node.addObject('RequiredPlugin', name='Sofa.Component.Visual')

    root_node.addObject('DefaultAnimationLoop')
    root_node.addObject('VisualStyle', displayFlags='hideVisualModels hideBehaviorModels showCollisionModels hideMappings showForceFields')

    m1 = root_node.addChild('M1')
    m1.addObject('EulerImplicitSolver')
    m1.addObject('CGLinearSolver', iterations="10", tolerance="1e-15", threshold="1e-15")
    m1.addObject('MechanicalObject', template=template)
    m1.addObject('UniformMass', totalMass=20 * size)
    m1.addObject('RegularGridTopology', nx="16", ny="16", nz=5*size+1, xmin="0", xmax="3", ymin="0", ymax="3", zmin="0", zmax=size)
    m1.addObject('FixedProjectiveConstraint', indices="0-255")
    m1.addObject('TetrahedronFEMForceField', name="FEM", youngModulus="24000", poissonRatio="0.3", method="large")
