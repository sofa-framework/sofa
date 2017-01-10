import Sofa
import sys
from SofaPython import script

############################################################################################
# this is a PythonScriptController example script
# with an instance callable in Python
############################################################################################


def createScene( root ):

    # here, you can notice how we can create a PythonScriptController with a true python variable
    # note how to give it optional extra arguments (@see additionalArguments)
    myController = MyControllerClass(root,"hello world controller", myArg = 'additionalArgument', myArg2 = 'additionalArgument2')

    # we have now access to its own member functions and variables
    myController.helloWorld()
    myController.myText = "hello world!"



class MyControllerClass(script.Controller):

   ### Overloaded PythonScriptController callbacks

    def createGraph(self, node):
       print "createGraph: myText ==",self.myText
       sys.stdout.flush()

       # here, you can notice how a controller can create another controller during its own creation
       RecursiveFibonacciControllerClass(node,"recursiveFibonacciController 0", f_1=0, f_2=1)


    def initGraph(self,node):
        print "initGraph: myText ==", self.myText
        sys.stdout.flush()


   ### Local variables / functions

    myText = "nothing to say"

    def helloWorld(self):
        print "helloWorld() function"
        sys.stdout.flush()


    ### to handle optional constructor arguments before createGraph
    def additionalArguments(self,kwarg):
        print "additionalArguments",kwarg
        self.myText = kwarg['myArg']




class RecursiveFibonacciControllerClass(script.Controller):

    nbInstances = 0

    def createGraph(self, node):
        RecursiveFibonacciControllerClass.nbInstances += 1
        f = self.f_1 + self.f_2
        print "Guys, I am crazy, I am creating myself recursively!"
        if RecursiveFibonacciControllerClass.nbInstances < 20:
            # here, you can notice how a controller can create another controller of the same class during its own creation, just awesome!
            self.recursive = RecursiveFibonacciControllerClass(node,"recursiveFibonacciController "+str(f), f_2=self.f_1, f_1=f )

    def additionalArguments(self,kwarg):
        self.f_1 = kwarg['f_1']
        self.f_2 = kwarg['f_2']