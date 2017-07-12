# coding=utf-8

import Sofa
import SofaPython.Tools
import SofaTest


def createScene(node):
    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')


class VerifController(SofaTest.Controller):

    def initGraph(self, node):

        name = u"aaéæœ®ßþ"
        mechanical_object = node.createObject("MechanicalObject", name=name)

        self.ASSERT(unicode(mechanical_object.name, "utf-8") == name, "test1")

        mechanical_object.name = name
        self.ASSERT(unicode(mechanical_object.name, "utf-8") == name, "test2")

        mechanical_object.name = u"ĳðð…"
        self.ASSERT(unicode(mechanical_object.name, "utf-8") == u"ĳðð…", "test3")

        self.ASSERT(isinstance(mechanical_object.name, str), "test4")

        Sofa.msg_info(mechanical_object.name)
        mechanical_object.name = mechanical_object.name

    def onEndAnimationStep(self, dt):
        self.sendSuccess()
