#include <fstream>

#include <sofa/helper/testing/BaseTest.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileSystem.h>

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager;

#include "PythonEnvironment.h"
#include "PythonTest.h"

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi=sofa::simpleapi;

namespace sofapython3
{

/// This function is used by gtest to print the content of the struct in a meaninfull way
void SOFAPYTHON3_API PrintTo(const sofapython3::PythonTestData& d, ::std::ostream *os)
{
    (*os) << d.filepath  ;
    (*os) << " with args {" ;
    for(auto& v : d.arguments)
    {
        (*os) << v << ", " ;
    }
    (*os) << "}";
}

///////////////////////// PythonTestData Definition  ///////////////////////////////////////////////
PythonTestData::PythonTestData(const std::string& filepath, const std::vector<std::string>& arguments ) :
    filepath(filepath), arguments(arguments) {}


///////////////////////// PythonTest Definition  //////////////////////////////////////////////////
PythonTest::PythonTest()
{
}

PythonTest::~PythonTest()
{
}

void PythonTest::run( const PythonTestData& data )
{
    msg_info("PythonTest") << "running " << data.filepath;
    PythonEnvironment::Init();
    {
        EXPECT_MSG_NOEMIT(Error);
        PythonEnvironment::setArguments(data.filepath, data.arguments);
        auto simulation = simpleapi::createSimulation();
        auto root = simulation->load(data.filepath.c_str());
    }
    PythonEnvironment::Release();
}

/// add a Python_test_data with given path
void PythonTestList::addTest( const std::string& filename,
                              const std::string& path,
                              const std::vector<std::string>& arguments)
{
    list.push_back( PythonTestData( filepath(path,filename), arguments ) );
}

void PythonTestList::addTestDir(const std::string& dir, const std::string& prefix)
{
    std::vector<std::string> files;
    sofa::helper::system::FileSystem::listDirectory(dir, files);

    for(const std::string& file : files)
    {
        if( sofa::helper::starts_with(prefix, file)
                && (sofa::helper::ends_with(".py", file) || sofa::helper::ends_with(".py3", file)))
        {
            addTest(file, dir);
        }
    }
}

} /// namespace sofapython3

