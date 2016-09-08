import Sofa
import sys
from SofaPython import script

############################################################################################
# this is a PythonScriptController example script
# with an instance callable in Python
############################################################################################


def createScene( root ):

    myController = MyControllerClass(root,"hello world controller", myArg = 'additionalArgument', myArg2 = 'additionalArgument2')
    myController.helloWorld()
    myController.myText = "hello world!"



class MyControllerClass(script.Controller):

   ### Overloaded PythonScriptController callbacks

    def createGraph(self, node):
       print "createGraph: myText ==",self.myText
       sys.stdout.flush()

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