#include <sofa/core/init.h>

#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sofa::core::init();
    int ret = RUN_ALL_TESTS();
    sofa::core::cleanup();
    return ret;
}
