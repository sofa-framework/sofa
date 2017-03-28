import SofaTest

def createScene(node):

    # use 'assert_true' when a failing test should terminate testing after this
    # time step
    SofaTest.assert_true(True, 'all went well')

    # use 'expect_true' when a failing test should not prevent the test from
    # running
    SofaTest.expect_true(True, 'iam ok with this')

    # your script should ***always*** call 'finish' once during its execution
    SofaTest.finish()

    # if a python error makes it to the toplevel, 'finish' is automatically
    # called for you
    
