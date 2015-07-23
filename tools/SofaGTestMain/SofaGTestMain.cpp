#include <sofa/simulation/tree/tree.h>
#ifdef SOFA_HAVE_DAG
#  include <sofa/simulation/graph/graph.h>
#endif

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sofa::simulation::tree::init();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::init();
#endif
    return RUN_ALL_TESTS();
}
