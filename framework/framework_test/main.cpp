#include <sofa/core/core.h>

#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sofa::core::init();
    return RUN_ALL_TESTS();
}
