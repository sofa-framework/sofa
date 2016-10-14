import Sofa
import SofaPython.Tools
import SofaTest

from SofaTest.Macro import *

import sys

def checkSysPathDuplicate():
#    for p in sys.path:
#        print p
    for p in sys.path:
#        print p
#        if (sys.path.count(p)>1):
#           Sofa.msg_info("Found duplicate path : "+p)
        if not EXPECT_EQ(1,sys.path.count(p),"sys.path.count("+p+")"):
            return False
    return True


def createScene(node):

    node.createObject('PythonScriptController', name='crtl1', filename=__file__, classname='VerifController')
    node.createObject('PythonScriptController', name='crtl2', filename=__file__, classname='VerifController')


class VerifController(SofaTest.Controller):

#    def initGraph(self, node):


    def onEndAnimationStep(self, dt):
        if not checkSysPathDuplicate():
            self.sendFailure()
        self.sendSuccess()
