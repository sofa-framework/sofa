#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/simulation/config.h> // #defines SOFA_HAVE_DAG (or not)
#ifdef SOFA_HAVE_DAG
#  include <SofaSimulationGraph/init.h>
#endif
#include <SofaSimulationTree/init.h>

#include <gtest/gtest.h>

using sofa::helper::system::PluginRepository;
using sofa::helper::system::DataRepository;
using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    sofa::simulation::tree::init();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::init();
#endif

    int ret =  RUN_ALL_TESTS();

#ifdef SOFA_HAVE_DAG
    sofa::simulation::tree::cleanup();
#endif
    sofa::simulation::graph::cleanup();

    return ret;
}
