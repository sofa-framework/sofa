#include <sofa/helper/system/PluginManager.h>

#include <vector>
#include <iostream>
#include <string>

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

#define BOOST_ALL_DYN_LINK

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp> // wchar_t must be define as built-in type on Windows in order to avoid undefined external symbol

#include <boost/regex.hpp>

using namespace boost::unit_test;

bool init_unit_test()
{

    int argc = boost::unit_test::framework::master_test_suite().argc;
    char** argv = boost::unit_test::framework::master_test_suite().argv;

    if( (argc == 2) && std::string( argv[1] ) == "-" )
    {
        // read from stdin

        std::string path;
        while( std::cin >> path )
		{
			std::cout << "Test : " << path << std::endl;
            plugin(path).load();
		}
    }
    else
    {
        if( argc==1 )
		{
            //std::cerr<<"Warning: runUnitTests expects a list of libraries to test in the command line, e.g.:  bin/runUnitTest lib/*_test.so " << std::endl;
			std::cout << "Gathering test units ..." << std::endl;

			boost::filesystem::path exePath = argv[0];
			boost::filesystem::path dirPath = exePath.parent_path();

			if(dirPath.empty())
				dirPath = "./";

			std::vector<boost::filesystem::path> files;
			std::copy(boost::filesystem::directory_iterator(dirPath), boost::filesystem::directory_iterator(), std::back_inserter(files));

			for(int i = 0; i < files.size(); ++i)
			{
				boost::filesystem::path filepath(files[i].filename());
				std::string filename(filepath.native().begin(), filepath.native().end());
				
				boost::filesystem::path extensionPath = filepath.extension();
				std::string extension(extensionPath.native().begin(), extensionPath.native().end());

				if(	std::string::npos != filename.find("_test") && std::string::npos == filename.find("boost") &&
					(0 == extension.compare(".dll") || 0 == extension.compare(".so") || 0 == extension.compare(".dynlib")))
				{
					std::cout << "Test : " << filename << std::endl;
					plugin(filename).load();
				}
			}
		}
		else
		{
			// read from argv
			for( int i = 1; i < argc; ++i)
			{
				std::cout << "Test : " << argv[i] << std::endl;
				plugin( argv[i] ).load();
			}
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

