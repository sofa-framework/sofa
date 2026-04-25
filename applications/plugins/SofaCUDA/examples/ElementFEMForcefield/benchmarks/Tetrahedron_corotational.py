import Sofa

import os
import numpy as np
from utilities import generate_regular_grid, hexa_to_tetra

g_grid_min_corner=(0, 6, -2)
g_grid_max_corner=(16, 10, 2)

g_fem_version = os.environ.get('FEM_VERSION', 'new') #either 'new' or 'legacy'
g_fem_template = os.environ.get('FEM_TEMPLATE', 'Vec3d')

# default is (76, 16, 16)
g_grid_nx = int(os.environ.get('NX', '76'))
g_grid_ny = int(os.environ.get('NY', '16'))
g_grid_nz = int(os.environ.get('NZ', '16'))

g_nb_steps = int(os.environ.get('NBSTEPS', '1000'))

def createScene(root_node):
    root_node.name = "root"
    root_node.gravity = (0, -9, 0)
    root_node.dt = 0.01

    plugin_node = root_node.addChild('Plugins')
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.Engine.Select")
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.LinearSolver.Iterative")
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.ODESolver.Backward")
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.StateContainer")
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.Topology.Container.Dynamic")
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.Topology.Container.Grid")
    plugin_node.addObject('RequiredPlugin', pluginName="Sofa.Component.Visual")
    plugin_node.addObject('RequiredPlugin', pluginName='Sofa.Component.Constraint.Projective') # Needed to use components [FixedProjectiveConstraint]  
    plugin_node.addObject('RequiredPlugin', pluginName='Sofa.Component.Mass') # Needed to use components [DiagonalMass]  
    plugin_node.addObject('RequiredPlugin', pluginName='Sofa.Component.SolidMechanics.FEM.Elastic') # Needed to use components [TetrahedronCorotationalFEMForceField]
    plugin_node.addObject('RequiredPlugin', pluginName='SofaCUDA.Component')
    plugin_node.addObject('VisualStyle', displayFlags="showBehaviorModels showForceFields")

    root_node.addObject('DefaultAnimationLoop')
    root_node.addObject('VisualStyle', displayFlags="showBehaviorModels showForceFields")

    grid_nodes, grid_hexa = generate_regular_grid(nx=g_grid_nx, ny=g_grid_ny, nz=g_grid_nz, min_corner=g_grid_min_corner, max_corner=g_grid_max_corner)
    grid_tetra = hexa_to_tetra(grid_hexa)
    
    tetrahedron_node = root_node.addChild('Tetrahedron')
    tetrahedron_node.addObject('EulerImplicitSolver', rayleighStiffness="0.1", rayleighMass="0.1")
    tetrahedron_node.addObject('CGLinearSolver', iterations="250", name="linear_solver", tolerance="1.0e-12", threshold="1.0e-12")
    tetrahedron_node.addObject('MechanicalObject', name="ms", template=g_fem_template, position=grid_nodes)
    tetrahedron_node.addObject('TetrahedronSetTopologyContainer', tetrahedra=grid_tetra)
    tetrahedron_node.addObject('DiagonalMass', totalMass="50.0")
    tetrahedron_node.addObject('BoxROI', name="boxroi1", box="-0.1 5 -3 0.1 11 3", drawBoxes="1")
    tetrahedron_node.addObject('FixedProjectiveConstraint', indices="@boxroi1.indices")
    if g_fem_version == "legacy":
        tetrahedron_node.addObject('TetrahedronFEMForceField', name="LegacyFEM", template=g_fem_template, youngModulus="4000", poissonRatio="0.3", method="large")
    if g_fem_version == "new":
        tetrahedron_node.addObject('TetrahedronCorotationalFEMForceField', name="NewFEM", template=g_fem_template, youngModulus="4000", poissonRatio="0.3")

def main():

    enable_gui = False

    try:
        import Sofa.Gui
        import SofaImGui
    except:
        enable_gui = False
    
    root = Sofa.Core.Node("root")
    createScene(root)
    
    Sofa.Simulation.initRoot(root)

    if enable_gui:
        Sofa.Gui.GUIManager.Init("myscene","imgui")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
    else:
        import time

        print(f"Running on {g_nb_steps} steps...")
        start_timer = time.time()

        for iteration in range(g_nb_steps):
            Sofa.Simulation.animate(root, root.dt.value)

        stop_timer = time.time()
        print(f"... Done.")
        print(f"{g_nb_steps} steps done in {stop_timer - start_timer:.3}s ({g_nb_steps/(stop_timer - start_timer):.5} fps).")


# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()
