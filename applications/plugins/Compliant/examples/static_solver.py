import xml.etree.ElementTree as ET
from itertools import izip

def requires(node, *names):

    for name in names:
        node.createObject( 'RequiredPlugin',
                           name = name,
                           pluginName = name )

def display_flags(node, **kwargs):

    items = [k + x for k, v in kwargs.iteritems() for x in v.split()]
    node.createObject('VisualStyle', displayFlags = ' '.join(items))

    
        

def load_xml(sofanode, xmlnode):
    '''load xml node under sofa node'''
    
    if xmlnode.tag == 'Node':

        name = xmlnode.attrib.get('name', '')
        sofachild = sofanode.createChild(name)
        
        for xmlchild in xmlnode:
            load_xml(sofachild, xmlchild)

        return sofachild
    else:
        return sofanode.createObject(xmlnode.tag, **xmlnode.attrib)
        
    
def beam(node, **kwargs):

    name = kwargs.get('name', 'beam')
    color = kwargs.get('color', '1 1 1')

    color = [ color + ' 0.3',
              color + ' 1']
    
    
    root = ET.parse('beam.xml').getroot()
    
    for model, c in izip(root.iter('VisualModel'), color):
        model.attrib['color'] = c
        
    res = load_xml(node, root)
    res.name = name
    return res
    

def createScene(node):

    node.gravity = '0 -10 0'
    node.dt = 1e-5
    
    display_flags(node, show = 'Behavior Visual',
                  hide = 'MechanicalMappings')
    
    requires(node, 'Flexible', 'Compliant')

    static = beam(node, name = 'static', color = '1 0.8 0.2')
    ode = static.createObject('CompliantStaticSolver',
                              line_search = True,
                              conjugate = True)
    ode.printLog = True
    
    # dynamic = beam(node, name = 'dynamic', color = '0.8 0.8 1')
    
    # dynamic.createObject('CompliantImplicitSolver')

    # dynamic.createObject('CgSolver',
    #                      iterations = 100,
    #                      precision= 1e-6)
