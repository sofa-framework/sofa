 #
 # Macro to mimic google test behavior
 # For now only EXPECT_* can be implemented (not EXPECT_* style)
 #

def EXPECT_TRUE(value, message=""):
    if not value:
        print "EXPECT_TRUE gets False - ", message
    return value

def EXPECT_FALSE(value, message=""):
    if value:
        print "EXPECT_FALSE gets True - ", message
    return not value
