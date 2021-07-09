import math
import numpy as np

#
# Macro to mimic google test behavior
# For now only EXPECT_* can be implemented (not EXPECT_* style)
#

def FAIL(message=""):
    print("Message:"+message)

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
    return EXPECT_FLOAT_EQ(expected, actual, message, rtol, atol)

def EXPECT_MAT_EQ(expected, actual, message="", rtol=1e-5, atol=1e-8):
    return EXPECT_FLOAT_EQ(expected, actual, message, rtol, atol)

def ASSERT_TRUE(value):
    """raise an exception if value is not True. The exception can then be catched
       at top level and converted into a Sofa.msg_error() which can be catched by the
       EXPECT_MSG_NOEMIT(Error) clause of BaseTest. So these failure are correctly reported in gtest
       (including line number).
    """
    if not value:
        message  = "Expecting 'True' while value is '"+str(value)+"' \n"
        raise Exception(message)

def ASSERT_FALSE(value):
        """raise an exception if value is not FALSE. The exception can then be catched
           at top level and converted into a Sofa.msg_error() which can be catched by the
           EXPECT_MSG_NOEMIT(Error) clause of BaseTest. So these failure are correctly reported in gtest
           (including line number).
        """
        if value:
            message  = "Expecting 'False' while value is '"+str(value)+"' \n"
            raise Exception(message)

def ASSERT_GT(value1, value2):
    """raise an exception if value1 <= value2. The exception can then be catched
       at top level and converted into a Sofa.msg_error() which can be catched by the
       EXPECT_MSG_NOEMIT(Error) clause of BaseTest. So these failure are correctly reported in gtest
       (including line number).
    """
    if value1 <= value2:
        message  = "Expecting value1 > value2 while got\n"
        message += "\t- value1: "+str(value1)+"\n"
        message += "\t- value2: "+str(value2)
        raise Exception(message)

def ASSERT_LT(value1, value2):
    """raise an exception if value1 >= value2. The exception can then be catched
       at top level and converted into a Sofa.msg_error() which can be catched by the
       EXPECT_MSG_NOEMIT(Error) clause of BaseTest. So these failure are correctly reported in gtest
       (including line number).
    """
    if value1 >= value2:
        message  = "Expecting value1 < value2 while got \n"
        message += "\t- value1: "+str(value1)+"\n"
        message += "\t- value2: "+str(value2)
        raise Exception(message)

def ASSERT_NEQ(value1, value2):
    """raise an exception if value1 & value2 are different. The exception can then be catched
       at top level and converted into a Sofa.msg_error() which can be catched by the
       EXPECT_MSG_NOEMIT(Error) clause of BaseTest. So these failure are correctly reported in gtest
       (including line number).
    """
    if value1 == value2:
        message  = "Expecting value1 & value2 to be different \n"
        message += "\t- value1: "+str(value1)+"\n"
        message += "\t- value2: "+str(value2)
        raise Exception(message)

def ASSERT_EQ(value1, value2):
    """raise an exception if value1 & value2 are different. The exception can then be catched
       at top level and converted into a Sofa.msg_error() which can be catched by the
       EXPECT_MSG_NOEMIT(Error) clause of BaseTest. So these failure are correctly reported in gtest
       (including line number).
    """
    if value1 != value2:
        message  = "Expecting value1 & value2 to be equal \n"
        message += "\t- value1: "+str(value1)+"\n"
        message += "\t- value2: "+str(value2)
        raise Exception(message)
