#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/simulation/tree/init.h>

#include <gtest/gtest.h>

using sofa::helper::system::PluginRepository;
using sofa::helper::system::DataRepository;
using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    sofa::simulation::tree::init();
    sofa::simulation::graph::init();

#ifdef WIN32
    const std::string pluginDirectory = Utils::getExecutableDirectory();
#else
    const std::string pluginDirectory = Utils::getSofaPathPrefix() + "/lib";
#endif
    sofa::helper::system::PluginRepository.addFirstPath(pluginDirectory);
    DataRepository.addFirstPath(std::string(SOFA_SRC_DIR) + "/share");

    int ret =  RUN_ALL_TESTS();

    sofa::simulation::graph::cleanup();
    sofa::simulation::tree::cleanup();

    return ret;
}
