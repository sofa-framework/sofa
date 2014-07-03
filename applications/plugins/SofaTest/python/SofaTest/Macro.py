import math
import numpy as np

#
# Macro to mimic google test behavior
# For now only EXPECT_* can be implemented (not EXPECT_* style)
#


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

def EXPECT_FLOAT_EQ(expected, actual, message="", rtol=1e-5, atol=1e-8):
    test = bool( np.allclose( expected, actual, rtol, atol ) )
    if not test:
        print "Value:", actual, "Expected:", expected, "-", message
    return test

def EXPECT_VEC_EQ(expected, actual, message="", rtol=1e-5, atol=1e-8):
    test = bool( np.allclose( expected, actual, rtol, atol ) )
    if not test:
        print "Value:", actual, "Expected:", expected, "-", message
    return test
