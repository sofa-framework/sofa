import numpy as np
np.set_string_function( lambda x: ' '.join( map(str, x)), repr = False )

def vec(*args):
    return np.array( args, float )


def require(node, plugin):
    return node.createObject('RequiredPlugin', pluginName = plugin)


def deriv_size( template ):
    if 'Vec' in template: return int(template[3])
    elif 'Rigid' in template: return 6
    # TODO more

def coord_size( template ):
    if 'Vec' in template: return int(template[3])
    elif 'Rigid' in template: return 7
    
def matrix_size(dofs):
    if type(dofs.velocity) == float: return 1
    return len(dofs.velocity) * len( dofs.velocity[0] )


def dofs(node, template, size = 1, **kwargs):

    kwargs.setdefault('position', size * np.zeros( coord_size(template)))
    kwargs.setdefault('velocity', size * np.zeros( deriv_size(template)))    
    
    return node.createObject('MechanicalObject',
                             name = 'dofs',
                             template = template,
                             **kwargs)

