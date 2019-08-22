
from itertools import izip
import scene
    
def chain(node, **kwargs):

    name = kwargs.get('name', 'chain')
    resolution = kwargs.get('n', 10)
    scale = kwargs.get('scale', "{} 0 0".format(resolution))
    stiffness = kwargs.get('stiffness', 1)
    mass = kwargs.get('mass', 1)

    color = kwargs.get('color', ".7 .7 .1 1")
    
    compliance = 1 / stiffness

    # TODO translation
    
    res = node.createChild(name)
    res.createObject('StringMeshCreator',
                     name="loader",
                     resolution = resolution,
                     scale = scale)
    
    res.createObject('MeshTopology', name="mesh", src="@loader" )
    res.createObject('MechanicalObject',
                      template="Vec3d",
                      name="dofs",
                      src="@loader",
                      showObject="1",
                      showObjectScale="0.05",
                      drawMode=1,
                      showColor=color )
    
    res.createObject('FixedConstraint', indices="0", drawSize=0.07 )
    res.createObject('UniformMass',  name="mass", mass=mass)
    
    sub = res.createChild("distance")
    sub.createObject('MechanicalObject',
                     template="Vec1d",
                     name="dofs" )
    sub.createObject('EdgeSetTopologyContainer',
                     edges="@../mesh.edges" )
    sub.createObject('DistanceMapping',
                     showObjectScale="0.02",
                     showColor= "1 1 1 1")
    sub.createObject('UniformCompliance',
                     name="ff",
                     template="Vec1d",
                     compliance = compliance,
                     isCompliance= False)
    
    return res


import numpy as np
import math

class Script(scene.Script):

    def onBeginAnimationStep(self, dt):
        pos = np.array(self.dofs.position)
        dist = np.array(self.distance.position)
        delta = dist.flatten() - self.expected
        error = math.sqrt( delta.dot(delta) )
        print 'error:', error



def createScene(node):

    node.gravity = '0 -1 0'
    node.dt = 1
    
    scene.display_flags(node, show = 'Behavior Visual',
                        hide = 'MechanicalMappings')
    
    scene.requires(node, 'Flexible', 'Compliant')

    n = 10
    c = chain(node, name = 'chain', n = n)
    
    ode = c.createObject('CompliantStaticSolver', ls_precision = 1e-14)
    
    # ode.printLog = True

    script = Script(node)
    script.dofs = c.getObject('dofs')

    script.distance = c.getChild('distance').getObject('dofs')
    script.expected = np.array( list(reversed(xrange(1, n))) )
