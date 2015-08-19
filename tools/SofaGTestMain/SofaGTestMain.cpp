#include <sofa/simulation/tree/init.h>
#include <sofa/simulation/graph/init.h>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sofa::simulation::tree::init();
    sofa::simulation::graph::init();
    int ret =  RUN_ALL_TESTS();
    sofa::simulation::graph::cleanup();
    sofa::simulation::tree::cleanup();
    return ret;
}
