import SofaPython
import sys
import os
import pslloader
import pslparserhjson
import pslast
from SofaTest.Macro import *

def testAstCompare():
    ast1 = [ ("a","b"), ("c",[("c1","cc")]), ("d",[]) ]
    ast2 = [ ("a","b"), ("c",[("c1","cc")]), ("d",[]) ]
    ast3 = [ ("a","b"), ("c",[("c1","cc")]), ("d",[""]) ]
    ast4 = [ ("a","b"), ("c",[("c1","cc"), ("","")]), ("d",[]) ]
    ok = True

    testlist = [(ast1, ast1, True), (ast1, ast2, True), (ast1, ast3, False), (ast1, ast4, False)]
    for astA, astB, res in testlist:
        s, m= pslast.compareAst(astA, astB)
        if s != res:
            FAIL("Failure on testAstCompare with "+str((astA, astB)) + " messsage:"+m)
            ok &= False
    return ok

def testAstLoading():
    ok = True
    testlist = [("data/test_ast.psl", "data/test_ast.pslx", True),
                ("data/test_ast.psl", "data/test_ast_broken.pslx", False)]
                #("data/test_caduceus.psl", "data/test_caduceus.pslx", True)]
    for astFileA, astFileB, res in testlist:
        ast1 = pslast.reorderAttributes(pslast.removeUnicode(pslloader.loadAst(astFileA)))
        ast2 = pslast.reorderAttributes(pslast.removeUnicode(pslloader.loadAst(astFileB)))
        s, m = pslast.compareAst(ast1, ast2)
        if s != res:
            FAIL("Failure on testAstLoading with "+str((astFileA, astFileB)) + "\n   messsage:"+m)
            ok &= False
    return ok


## the python test MUST have a "run" function with no arguments that returns the test result
def run():
    try:
        dirname=os.path.dirname(os.path.abspath(__file__))
        with pslloader.SetPath(dirname):
            ok = True
            if not testAstCompare():
                ok = False

            if not testAstLoading():
                ok = False

            return ok

        return False

    except Exception, e:
        SofaPython.sendMessageFromException(e)
        return False
