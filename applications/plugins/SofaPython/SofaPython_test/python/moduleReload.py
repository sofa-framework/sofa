import Sofa
import SofaTest

# this module has been unloaded so the random number
# generated in this module must have changed
import myrandom
currentNumber = myrandom.random_number

try:
    previousNumber
except:
    previousNumber = currentNumber-1


def createScene(node):

    node.animate = True

    node.createObject('PythonScriptController', classname='VerifController')



class VerifController(SofaTest.Controller):

    def createGraph(self,node):
        self.node=node

    def onEndAnimationStep(self, dt):

        global previousNumber

        # print "onEndAnimationStep",counter,willUnloadModules,unloadingModules,previousNumber,currentNumber; sys.stdout.flush()

        # ensure the new myrandom module import has generated a new number
        self.ASSERT( previousNumber!=currentNumber, "should be different")

        previousNumber = currentNumber

        self.sendSuccess()

        self.node.animate = False
