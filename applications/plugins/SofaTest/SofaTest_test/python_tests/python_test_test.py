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
  return EXPECT_TRUE(isNull( int(sys.argv[1]) ), "isNull") and EXPECT_FALSE(isNull( int(sys.argv[2]) ), "isNull" )
