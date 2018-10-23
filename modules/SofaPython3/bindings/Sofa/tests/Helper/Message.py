# coding: utf8

import Sofa
import unittest

class Test(unittest.TestCase):
    def test_messages(self):
        """Test that the message are correctly sended and does not generates exceptions"""
        for fname in ["msg_info", "msg_warning", "msg_deprecated"]:
            f = getattr(Sofa.Helper, fname)
            f("Simple message")
            f("Emitter", "Simple message")
            f("Simple message with attached source info", "sourcefile.py", 10)
            f(Sofa.Node("node"), "Simple message to an object")
            f(Sofa.Node("node"), "Simple message to an object with attached source info", "sourcefile.py", 10)

            
def runTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

def createScene(rootNode):
        runTests()
