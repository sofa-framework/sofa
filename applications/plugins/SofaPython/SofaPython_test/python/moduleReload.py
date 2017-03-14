
try:
    counter += 1
    unloadingModules = willUnloadModules
except:
    counter = 0
    unloadingModules = False

willUnloadModules = counter%3==0

# # forcing the already loaded modules to be unloaded
# # once in a while
# if unloadingModules:
#     SofaPython.unloadModules()


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

import numpy


def createScene(node):


    node.animate = True

    # just to call numpy code which has a particular import
    x = numpy.array([[currentNumber, currentNumber, currentNumber], [currentNumber, currentNumber, currentNumber]])


    if willUnloadModules:
        node.createObject('PythonModuleReload', name='force_module_reload')

    node.createObject('PythonScriptController', classname='VerifController')



class VerifController(SofaTest.Controller):

    def createGraph(self,node):
        self.node=node

    def onEndAnimationStep(self, dt):

        global previousNumber

        # print "onEndAnimationStep",counter,willUnloadModules,unloadingModules,previousNumber,currentNumber; sys.stdout.flush()

        if unloadingModules:
            # ensure the new myrandom module import has generated a new number
            self.ASSERT( previousNumber!=currentNumber, "should be different")
        else:
            self.ASSERT( previousNumber==currentNumber, "should be similar")


        previousNumber = currentNumber

        self.sendSuccess()

        self.node.animate = False
