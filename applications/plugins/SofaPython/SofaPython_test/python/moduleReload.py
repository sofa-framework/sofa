
try:
    counter += 1
except:
    counter = 0

unloadingModules = counter%3==0


# forcing the already loaded modules to be unloaded
# once if a while
if unloadingModules:
    SofaPython.unloadModules()


import Sofa
import SofaTest

# this module have been unloaded so the random number
# generated in this module must have changed
import myrandom
currentNumber = myrandom.random_number

try:
    previousNumber
except:
    previousNumber = currentNumber-1 if unloadingModules else currentNumber




def createScene(node):

    node.createObject('PythonScriptController', classname='VerifController')




class VerifController(SofaTest.Controller):

    def onEndAnimationStep(self, dt):

        global previousNumber
        try:
            if unloadingModules:
                # ensure the new myrandom module import has generated a new number
                self.ASSERT( previousNumber!=currentNumber, "should be different")
            else:
                self.ASSERT( previousNumber==currentNumber, "should be similar")
        except:
            pass

        previousNumber = currentNumber

        self.sendSuccess()
