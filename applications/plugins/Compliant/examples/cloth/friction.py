

import scene


def createScene(node):
    node.gravity = '0 -10 0'
    node.dt = 1e-2
    
    scene.display_flags(node, show = 'Behavior Visual CollisionModels',
                  hide = 'MechanicalMappings')
    
    scene.requires(node, 'Flexible', 'Compliant')

    scene.contacts(node,
                   response = 'FrictionCompliantContact',                 
                   # response = 'CompliantContact',
                   )

    c = cloth(node)
    b = ball(node)
    
    ode = node.createObject('CompliantImplicitSolver',
                            stabilization = 0,
                            neglecting_compliance_forces_in_geometric_stiffness = False)
    
    num = node.createObject('ModulusSolver',
                            iterations = 15,
                            precision = 1e-6,
                            anderson = 4)

    # num = node.createObject('SequentialSolver',
    #                         iterations = 10,
    #                         precision = 1e-6,
    #                         anderson = 4)
    

def ball(node, **kwargs):
    root = scene.xml_load('ball.xml')

    name = kwargs.get('name', 'ball')
    res = scene.xml_insert(node, root)
    res.name = name

    return res


def cloth(node, **kwargs):

    root = scene.xml_load('cloth.xml')

    name = kwargs.get('name', 'cloth')

    res = scene.xml_insert(node, root)
    res.name = name

    return res
