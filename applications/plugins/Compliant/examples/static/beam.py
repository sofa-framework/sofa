
from itertools import izip
import scene
    
def beam(node, **kwargs):

    name = kwargs.get('name', 'beam')
    color = kwargs.get('color', '1 1 1')

    color = [ color + ' 0.3',
              color + ' 1']
    
    
    root = scene.xml_load('beam.xml')
    
    for model, c in izip(root.iter('VisualModel'), color):
        model.attrib['color'] = c
        
    res = scene.xml_insert(node, root)
    res.name = name
    return res
    

def createScene(node):

    node.gravity = '0 -10 0'
    node.dt = 1e-5
    
    scene.display_flags(node, show = 'Behavior Visual',
                  hide = 'MechanicalMappings')
    
    scene.requires(node, 'Flexible', 'Compliant')

    static = beam(node, name = 'static', color = '1 0.8 0.2')

    ode = static.createObject('CompliantStaticSolver',
                              ls_iterations = 10,
                              ls_precision = 1e-5,
                              line_search = 2,
                              conjugate = True)
    ode.printLog = True
    
    # dynamic = beam(node, name = 'dynamic', color = '0.8 0.8 1')
    
    # dynamic.createObject('CompliantImplicitSolver')

    # dynamic.createObject('CgSolver',
    #                      iterations = 100,
    #                      precision= 1e-6)
