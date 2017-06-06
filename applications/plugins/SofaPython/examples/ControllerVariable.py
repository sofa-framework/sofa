from __future__ import print_function

import Sofa
import sys
from SofaPython import script


############################################################################################
# this is a PythonScriptController example script
# with an instance callable in Python
############################################################################################


def createScene( root ):

    # here, you can notice how we can create a PythonScriptController with a
    # true python variable note how to give it optional extra arguments (@see
    # additionalArguments)
    myController = MyControllerClass(root, "hello world controller",
                                     myArg = 'additionalArgument',
                                     myArg2 = 'additionalArgument2')

    # we have now access to its own member functions and variables
    myController.helloWorld()
    myController.myText = "hello world!"



class MyControllerClass(Sofa.PythonScriptController):

   ### Overloaded PythonScriptController callbacks
    def __init__(self, node, name, *args, **kwargs):
       print("createGraph: myText ==", self.myText)

       # note: this member aliases the sofa component `name` data, so you can't
       # put anything in here (in this case, strings only)
       self.name = name

       # this one only exists on the python side (no such data exists in the c++
       # object)
       self.myText = kwargs['myArg']
       
       # here, you can notice how a controller can create another controller
       # during its own creation
       sub = RecursiveFibonacciControllerClass(node, "recursiveFibonacciController 0", f_1 = 0, f_2 = 1)
       
    def initGraph(self,node):
        print("initGraph: myText ==", self.myText)


    ### class variables / functions
    myText = "nothing to say"
    
    def helloWorld(self):
        print("helloWorld() function")




class RecursiveFibonacciControllerClass(Sofa.PythonScriptController):

    nbInstances = 0

    def __init__(self, node, name, *args, **kwargs):

        RecursiveFibonacciControllerClass.nbInstances += 1

        self.name = name
        
        self.f_1 = kwargs['f_1']
        self.f_2 = kwargs['f_2']
        
        f = self.f_1 + self.f_2

        # lolwat
        print("Guys, I am crazy, I am creating myself recursively!")
        
        if RecursiveFibonacciControllerClass.nbInstances < 20:
            # here, you can notice how a controller can create another
            # controller of the same class during its own creation, just
            # awesome!
            self.recursive = RecursiveFibonacciControllerClass(node, "recursiveFibonacciController " + str(f),
                                                               f_2 = self.f_1, f_1 = f)
            

