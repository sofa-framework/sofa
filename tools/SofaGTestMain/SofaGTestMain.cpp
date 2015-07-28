#include <sofa/simulation/tree/init.h>
#ifdef SOFA_HAVE_DAG
#  include <sofa/simulation/graph/init.h>
#endif

#include <gtest/gtest.h>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sofa::simulation::tree::init();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::init();
#endif
    int ret =  RUN_ALL_TESTS();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::cleanup();
#endif
    sofa::simulation::tree::cleanup();
    return ret;
}
