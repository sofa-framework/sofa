import sys

from SofaTest.Macro import *

# arguments are an option in python tests
if len(sys.argv) != 3 :
    print "ERROR: wrong number of arguments"
  
# a pure python function to test
def isNull( x ):
    return x==0

# the python test MUST have a "run" function with no arguments that returns the test result
def run():
    # here it tests if the first argument is null and the second is not
    ok = True
    ok &= EXPECT_TRUE(isNull( int(sys.argv[1]) ), "isNull")
    ok &= EXPECT_FALSE(isNull( int(sys.argv[2]) ), "isNull")
    ok &= EXPECT_EQ("toto","toto", "EXPECT_EQ")
    # a very small value
    e = 1e-8
    ok &= EXPECT_FLOAT_EQ(1.0, 1.0+e, "EXPECT_FLOAT_EQ")
    ok &= EXPECT_VEC_EQ([1.+e, 2.-e, 3.], [1., 2., 3.], "EXPECT_VEC_EQ")
    return ok
