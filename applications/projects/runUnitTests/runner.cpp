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


loader loader::instance;

// this will generate a main function for running tests
#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>


