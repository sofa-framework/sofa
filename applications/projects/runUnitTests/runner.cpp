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

	    // TODO is there a reason why we should init the plugin manager
	    // on each plugin load ?
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




#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>


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
        if( argc==1 )
            std::cerr<<"Warning: runUnitTests expects a list of libraries to test in the command line, e.g.:  bin/runUnitTest lib/*_test.so " << std::endl;

        // read from argv
        for( int i = 1; i < argc; ++i)
        {
            plugin( argv[i] ).load();
        }

    }

    framework::master_test_suite().p_name.value = "SOFA Test Suite";

    return true;
}

int BOOST_TEST_CALL_DECL
main( int argc, char* argv[] )
{
    return ::boost::unit_test::unit_test_main( &init_unit_test, argc, argv );
}

