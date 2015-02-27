


import patch

type_dict = {}

def cast( obj ):
    '''cast an object with its class, as decorated by sofa_object'''

    # TODO fancy template wizardry
    cls = type_dict.get(obj.getClassName(), {}).get(obj.getTemplateName(), None)

    if not cls: return obj
    
    return patch.instance(obj, cls)


def extends(base):
    '''decorator for SofaPython class extensions'''
    def res(cls):
        patch.class_dict( base ).update( cls.__dict__ )
        return cls

    return res

def sofa_class(classname,
               template = ''):
    '''decorator for casting classes'''
    
    def res( cls ):
        type_dict.setdefault(classname, {})[template] = cls
        return cls

    return res
    
# actual wrapping starts here
# should probably go somewhere else

import Sofa

@extends(Sofa.Node)
class Node:
    _createObject = Sofa.Node.createObject
    _getObject = Sofa.Node.getObject

    def createObject(self, typename, **kwargs):
        return cast( Node._createObject(self, typename, **kwargs) )

    def getObject(self, name):
        return cast( Node._getObject(self, name) )
    


import numpy as np

# magic function to get proper string output
np.set_string_function( lambda x: ' '.join( map(str, x)), repr = False )
    
@sofa_class('MechanicalObject', 'Rigid3f')
@sofa_class('MechanicalObject', 'Rigid3d')
class RigidObject(Sofa.BaseMechanicalState):

    @property
    def center(self):
        return np.array(self.position[0][:3])

    @center.setter
    def center(self, value):
        self.position = str(np.hstack( (value, self.position[0][3:])) )
   
