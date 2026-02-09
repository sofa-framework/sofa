# Required import for python
import Sofa
import SofaRuntime


def main():
    # Make sure to load all necessary libraries
    SofaRuntime.importPlugin("Sofa.Component.StateContainer")

    # Call the above function to create the scene graph
    root = Sofa.Core.Node("root")
    createScene(root)

    # Once defined, initialization of the scene graph
    Sofa.Simulation.initRoot(root)

    # Run the simulation for 10 steps
    for iteration in range(10):
        print(f'Iteration #{iteration}')
        Sofa.Simulation.animate(root, root.dt.value)

    print("Simulation made 10 time steps. Done")


# Function called when the scene graph is being created
def createScene(root):

    root.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')

    # Scene must now include a AnimationLoop
    root.addObject('DefaultAnimationLoop')

    # Add new nodes and objects in the scene
    node1 = root.addChild("Node1")    
    node1.addObject("MechanicalObject", template="Rigid3d", position="0 0 0   0 0 0 1", showObject="1")

    node2 = root.addChild("Node2")
    node2.addObject("EulerImplicitSolver")
    node2.addObject("CGLinearSolver", iterations="100", tolerance="1e-3", threshold="1e-3")
    node1.addChild(node2)

    node3 = root.addChild("Node3")
    node3.addObject("MechanicalObject", template="Rigid3d", position="0 0 0   0 0 0 1", showObject="1")
    node3.addChild(node2)

    return root


# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()
