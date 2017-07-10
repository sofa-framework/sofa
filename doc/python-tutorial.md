
Sofa Tutorial
===========
Introduction
-

This tutorial will give you the tools to get started with Sofa. It is based on few examples that you can find on Sofa repository on [GitHub](https://github.com/sofa-framework/sofa.git/).
Here, we use Python scripts to implement the scenes that we open with Sofa.

> In Sofa, a scene is represented as a graph, and more specially as a tree. Every object of the scene is a node of this tree. Each object can be characterized by components, describing its nature, its behavior and its constraints.

Step by step
-


 #### Preliminaries

1. Edit a new Python script : `firstscene.py`
2. Open Sofa and open `firstscene.py`

From now, every time you want to update your scene, you need to save your Python script and reload the  scene in Sofa ( *File > Reload* or *Ctrl+R*)

 #### Create a scene

In the following steps, we will use an example as a guide line. The scene consists of representing two particles linked by a spring. These two particles have a mass ($$m$$) and the stiffness of the spring is $$k$$. One of the particles is fixed, the other is mobile.

1. Create a function `createScene(node)`
        - This function is recognized by Sofa to create a scene
        - Declare every component of your scene in this function
        - `node` parameter represents the root of your graph
2. Add a particle
        - First, create a new node (child of the root)
        - Then, specify its nature (here, `MechanicalObject`).
        - Add a mass to the particle
        - New components are visible in the description of the graph in Sofa
3. Draw the particle

Here is the code for the previous steps :

```python
    def createScene(node) :
        #Creation of the particle (child of the root of the graph)
        p = node.createChild('particle')

        #Creation of the degrees of freedom of the new particle
        #Here, we are in 3D, so we use the template 'Vec3' that create vectors in 3 dimensions.
        dofs = p.createObject('MechanicalObject', template='Vec3', size=1, name='dofs')

        #Add the mass to the particle
        mass = p.createObject('UniformMass', template = 'Vec3', mass = 1.0)

        #Draw the particle
        dofs.showObject = True
        dofs.drawMode = 2
```

Now we want to create another particle and model a spring attraction between both particles.
To avoid code duplication, we create a function `create_particle`. When both of the particles are created, we fix the first one.

The two particles are linked by a spring. To model this link, we need to create a new node that represents the spring attraction.
This node is the child of the two particles. It is composed of a MechanicalObject, a DifferenceMultiMapping and a UniformStiffness.

To animate the scene, you need to use two solvers `EulerImplicitSolver` and `CGLinearSolver`.

Here is the final code for the simulation of two particles linked by a spring.

```python
def create_particle(node, name, mass, position) :
    #Creation of the particle (child of the root of the graph)
    p = node.createChild(name)

    #Creation of the degrees of freedom of the new particle
    #Here, we are in 3D, so we use the template 'Vec3' that create vectors in 3 dimensions.
    dofs = p.createObject('MechanicalObject', template='Vec3', size=1, name='dofs')

    #Add the mass to the particle
    mass = p.createObject('UniformMass', template = 'Vec3', totalMass = mass)

    #Set the position of the particle
    dofs.position = position

    #Draw the particle
    dofs.showObject = True
    dofs.drawMode = 2

    return p

def createScene(node) :

    #Server for ode
    ode_solver = node.createObject('EulerImplicitSolver')

    #Server for linear systems
    num_solver = node.createObject('CGLinearSolver')

    #Creation of 2 particles
    p1 = create_particle(node, 'particule1', 1.0, [-1.0,0,0])
    p2 = create_particle(node, 'particule2', 1.0, [1.0,0,0])

    #Fix one of them
    p2.createObject('FixedConstraint', fixAll = True)

    # Creation of a new node to represent the difference as child of a one particule
    diff = p1.createChild('difference')

    # Creation of the degrees of freedom of the difference
    dofs = diff.createObject('MechanicalObject', template = 'Vec3', size = 1)

    # Creation of the mapping from the two particles
    mapping = diff.createObject('DifferenceMultiMapping', template = 'Vec3,Vec3',
                                input = '@/particule1/dofs @/particule2/dofs',
                                output = dofs.getLinkPath())

    # note: you can get the path of an object in the graph from getLinkPath()

    #Link the new node as a child of the second particule
    p2.addChild(diff)

    #Fix the stiffness of the spring
    diff.createObject('UniformStiffness', template = 'Vec3', stiffness = 1e1)


```

#### Example
In Sofa repository, you can find many scene examples. Some of them are tutorials to assimilate the basic skills to work with Sofa. Here you can find some support about the pendulum basic tutorial.

>Note : You can find the .scn files in sofa > examples > Tutorials > Basic


##### Tutorial Basic Pendulum #
---


```python
import Sofa

def createScene(node):

    #Node property
    node.gravity = [0, 0, 0]
    node.dt = 0.1

    #Drawing settings
    node.createObject('VisualStyle', displayFlags = 'showBehavior', name = 'visualStyle1')

    #Solvers :
    ode_solver = node.createObject('EulerImplicitSolver')
    num_solver = node.createObject('CGLinearSolver', iterations = 25,
                                    tolerance = 1e-5, threshold=1e-5)

    #Particules settings
    particule = node.createObject('MechanicalObject', name = 'Particles',
                                   template = 'Vec3d', size = 2,
                                   position = "0 0 0 0 0 1", velocity = "0 0 0 0 1 0")

    #Mass settings
    masse= node.createObject('UniformMass', name = 'Mass',
                              template = 'Vec3d', totalMass = 1,  )

    # Constraint settings
    node.createObject('FixedConstraint', indices = 0, name = 'fixedConstraint1')

    # Spring force field settings
    node.createObject('StiffSpringForceField', name = 'Springs',
                       stiffness = 100, damping = 1,  spring = '0 1 10 1 1')

    # Sphere settings
    node.createObject('TSphereModel', radius = 0.1, name = 'tSphereModel1')

```
