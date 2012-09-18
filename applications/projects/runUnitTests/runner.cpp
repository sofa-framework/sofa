#include <sofa/helper/system/PluginManager.h>

#include <vector>
#include <iostream>

struct plugin
{
    std::string path;

    plugin( const std::string& path ) : path(path) { }

    void load()
    {
        std::cout << "loading " << path << "... " << std::flush;
        if (sofa::helper::system::PluginManager::getInstance().loadPlugin( path ))
        {
            std::cout << "done." << std::endl;

            sofa::helper::system::PluginManager::getInstance().init();
        }
    }
};

struct loader
{

    // read plugin names from std::cin until eof
    loader()
    {

        std::string path;
        while( std::cin >> path ) plugin( path ).load();

    };

    static loader instance;
};


#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>

using namespace boost::unit_test;

bool init_unit_test()
{

    int argc = boost::unit_test::framework::master_test_suite().argc;
    char** argv = boost::unit_test::framework::master_test_suite().argv;

    if( (argc == 2) && std::string( argv[1] ) == "-" )
    {
        // read from stdin

        std::string path;
        while( std::cin >> path ) plugin( path ).load();
    }
    else
    {
        // read from argv

        for( unsigned i = 1; i < argc; ++i)
        {
            plugin( argv[i] ).load();
        }

    }

    framework::master_test_suite().p_name.value = "SOFA Test Suite";

    return true;
}

