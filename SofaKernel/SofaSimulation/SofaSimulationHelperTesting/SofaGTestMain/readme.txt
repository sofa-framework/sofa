SofaGTestMain
*************

The SofaGTestMain library contains only a main function that initializes Sofa
and runs a Google Test test suite.  (It replaces the gtest_main library provided
with Google Test, that provides a that contains a minimal main() function.)
Each test executable is simply linked against it, unless it also needs to run
its own main() function.
