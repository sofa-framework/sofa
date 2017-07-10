import SofaPython.Tools

from SofaTest.Macro import *

def run():
    ok=True
    l = [1,2,3,4]
    ok&=EXPECT_EQ(repr(l), SofaPython.Tools.listToStr(l) )
    return ok

