# -*- coding: utf-8 -*-

class Wrapper(object):
    '''
    Args:
        node : the current node we are working on

        attachedFunction : the function that will be called at each object creation 
                           to do some stuff replace/insert/delete ...

                           This function will take as arguments the node, the type of the object to create,
                           datacache & also the current arguments of the object .

                           This function as to return a tuple containing parameters 
                           of the object we want to create (ie: his type and a dictionary 
                           with all the other arguments) or None 

        datacache : the data we will use in our attachedFunction as parameters or else

    '''
    
    def __init__(self, node, attachedFunction, datacache):
        self.node = node
        self.attachedFunction = attachedFunction
        self.datacache = datacache

    def createObject(self, type, **kwargs):

        objectArg = self.attachedFunction(self.node,type,self.datacache,kwargs) 
        # objectArg as to contain (newType, **newKwargs)

        if objectArg == None :
            return self.node.createObject(type, **kwargs)
        else : 
            return self.node.createObject(objectArg[0], **objectArg[1])

    def createChild(self, name):
        return Wrapper(self.node.createChild(name), self.attachedFunction ,self.datacache)
    
    def __getattr__(self, value):
        return self.node.__getattribute__(value)

### This functions are just an example on how to use the Wrapper class. 

# !!    Doesn't Work Any More With Current MainHeader Implementation    !!
#       A PR has to be made to make the Wrapper inherit from Sofa.Node

# # My param
# myNewParam = { "SofaPython" : 'SofaMiscCollision',
#                "SofaMiscCollision" : "SoftRobots",
#                "SoftRobots" : "SofaPython" }

# # My function that will replace a riquered plugin by another
# def myAttachedFunction(node,type,newParam,initialParam):
#     if str(type) == 'RequiredPlugin':
#         for key in newParam:
#             if initialParam['name'] == key:
#                 initialParam['name'] = newParam[key] 
#                 return type , initialParam

#     return None

# import mainheader

# # My new scene
# def createScene(rootNode):
#     ## Call the old scene creation. 
#     mainheader.createScene(Wrapper(rootNode, myAttachedFunction, myNewParam))
