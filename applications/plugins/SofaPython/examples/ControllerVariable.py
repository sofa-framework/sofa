import Sofa
import sys
from SofaPython import script

############################################################################################
# this is a PythonScriptController example script
# with an instance callable in Python
############################################################################################


def createScene( root ):
    
    myController = MyControllerClass(root,"toto")
    myController.helloWorld()
    myController.myText = "hello world!"
    
  
  
class MyControllerClass(script.Controller):
                               
   ### Overloaded PythonScriptController callbacks
   
        def onBeginAnimationStep(self,dt):
            print self.myText
            sys.stdout.flush()
            
   
   ### Local variables / functions
   
        myText = "nothing to say"
                
        def helloWorld(self):
            print "helloWorld() function"
            sys.stdout.flush()
