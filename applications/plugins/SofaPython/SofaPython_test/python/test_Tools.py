import SofaPython.Tools

from SofaTest.Macro import *

def run():
    ok=True
    l = [1,2,3,4]
    ok&=EXPECT_EQ("1 2 3 4", SofaPython.Tools.listToStr(l) )
    return ok

