import math
import numpy
import numpy.linalg

#
# Macro to mimic google test behavior
# For now only EXPECT_* can be implemented (not EXPECT_* style)
#

EPSILON = 1e-6

def EXPECT_TRUE(value, message=""):
    if not value:
        print "Value:", value, "Expected: True", message
    return value

def EXPECT_FALSE(value, message=""):
    if value:
        print "Value:", value, "Expected: False", message
    return not value

def EXPECT_EQ(expected, actual, message=""):
    if not expected==actual:
        print "Value:", actual, "Expected:", expected, "-", message
    return expected==actual

def EXPECT_FLOAT_EQ(expected, actual, message=""):
    test = bool(math.fabs(expected-actual)<EPSILON)
    if not test:
        print "Value:", actual, "Expected:", expected, "-", message
    return test

def EXPECT_VEC_EQ(expected, actual, message=""):
    test = bool(numpy.linalg.norm(numpy.array(expected)-numpy.array(actual))<EPSILON)
    if not test:
        print "Value:", actual, "Expected:", expected, "-", message
    return test
