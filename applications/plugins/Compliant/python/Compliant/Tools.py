# mostly helper functions
#
# author: maxime.tournier@inria.fr
#
#

import os

# concatenate lists for use with data. TODO: consider using
# Vec.Proxy.__str__ instead as it's much more powerful
def cat(x):
    return ' '.join(map(str, x))


# absolute path from filename
def path( name ):
    return os.path.dirname( os.path.abspath( name ) )


# reasonable standard scene
def scene(node):
    
    node.createObject('RequiredPlugin', pluginName = "Compliant" )
    
    #node.createObject('CompliantAttachButtonSetting') # Compliant friendly mouse picking but not working yet with rigid...

    node.dt = 0.01
    node.gravity = '0 -9.81 0'

    node.createObject('DefaultPipeline', name = 'pipeline')
    node.createObject('BruteForceDetection', name = 'detection')
    
    proximity = node.createObject('NewProximityIntersection', name = 'proximity' )
    
    manager = node.createObject('DefaultContactManager',
                                name = 'manager',
                                response = "FrictionCompliantContact",
                                responseParams = "mu=1" )
    
    style = node.createObject('VisualStyle', 
                              name = 'style',
                              displayFlags='hideBehaviorModels hideCollisionModels hideMappings hideForceFields')

    ode = node.createObject('CompliantImplicitSolver',
                            name='ode' )

    # TODO is this needed ?
    group = node.createObject('CollisionGroup', 
                              name = 'group' )

    return node.createChild('scene') 


def node_depth(node):
    parents = node.getParents()
    if len(parents) == 0: return 0
    else: return 1 + node_depth( parents[0] )
    

# path from root
# DEPRECATED : use binded BaseNode::getPathName() instead
def node_path_abs(node):
    parents = node.getParents()
    if len(parents) == 0: return ''
    else: return node_path_abs(parents[0]) + '/' + node.name
    

def node_path_rel(src, dst):

    return ('../' * node_depth(src)) + node_path_abs(dst)
    

# recursive find (nodes only, depth-first)
def find(root, name):
    
    if root.name == name:
        return root
        
    for c in root.getChildren():
        res = find(c, name)
        if res != None: 
            return res

    return None



    
    
