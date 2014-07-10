import SofaPython.Tools

from SofaTest.Macro import *

def run():
    ok=True
    ok&=EXPECT_EQ("1 2 3 4", SofaPython.Tools.listToStr([1,2,3,4]) )
    return ok

