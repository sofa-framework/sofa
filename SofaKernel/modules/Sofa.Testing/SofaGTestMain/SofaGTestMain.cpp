#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/simulation/config.h> // #defines SOFA_HAVE_DAG (or not)
#include <SofaSimulationGraph/init.h>

#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    sofa::simulation::graph::init();
    int ret =  RUN_ALL_TESTS();
    sofa::simulation::graph::cleanup();

    return ret;
}
