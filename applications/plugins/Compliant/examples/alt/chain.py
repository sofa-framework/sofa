
from itertools import izip
import scene
    
def chain(node, **kwargs):

    name = kwargs.get('name', 'chain')
    resolution = kwargs.get('n', 10)
    scale = kwargs.get('scale', "{} 0 0".format(resolution))
    stiffness = kwargs.get('stiffness', 1)
    mass = kwargs.get('mass', 1)
    use_compliance = kwargs.get('use_compliance', True)
    
    color = kwargs.get('color', ".7 .7 .1 1")
    
    compliance = kwargs.get('compliance', 1 / stiffness)

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

    if use_compliance:
        sub.createObject('UniformCompliance',
                         name="ff",
                         template="Vec1d",
                         compliance = compliance,
                         isCompliance= True)
    else:
        lagrange = sub.createChild("lagrange")
        dofs = lagrange.createObject("MechanicalObject", template = "Vec1d", name = 'dofs')
        dofs.position = ' '.join( ['0'] * (resolution - 1) )

        if compliance > 0:
            lagrange.createObject("UniformCompliance",
                                  template = "Vec1d",
                                  compliance = 1/compliance)

        constraint = sub.createChild('constraint')
        constraint.createObject('MechanicalObject', name = 'dofs', template = 'Vec1d')
        constraint.createObject('PairingMultiMapping', template = 'Vec1d,Vec1d',
                                input = '@../lagrange/dofs @../dofs',
                                output = '@dofs')

        constraint.createObject('PotentialEnergy', sign="1")
    
    return res





def createScene(node):

    node.gravity = '0 -1 0'
    node.dt = 0.1
    
    scene.display_flags(node, show = 'Behavior Visual',
                        hide = 'MechanicalMappings')
    
    scene.requires(node, 'Flexible', 'Compliant')

    
    n = 10

    for x in [False]:
        color = "1 0 0 1" if x else "0 1 0 1"
        c = chain(node, name = 'chain-{}'.format(x),
                  n = n,
                  compliance = 0,
                  use_compliance = x,
                  color = color)

        ode = c.createObject('CompliantImplicitSolver', stabilization = 0)
        num = c.createObject('MinresSolver', iterations = 100, precision = 1e-14)
        
