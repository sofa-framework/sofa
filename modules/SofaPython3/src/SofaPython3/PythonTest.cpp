#include <fstream>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileSystem.h>

#include "PythonTest.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager;

namespace sofapython3
{

py::scoped_interpreter pyinterp;

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

///////////////////////// PythonTest Definition  //////////////////////////////////////////////////
PythonTest::PythonTest()
{
    //sofa::helper::system::PluginManager::getInstance().loadPlugin("SofaPython3");

    std::cout << "TEST "<< std::endl;

    //py::scoped_interpreter guard{};
    std::cout << "TEST "<< std::endl;
    //py::module::import("SofaRuntime");

    py::exec(R"(
            kwargs = dict(name="World", number=42)
            message = "Hello, {name}! The answer is {number}".format(**kwargs)
            print(message)
        )");

    std::cout << "TEST "<< std::endl;
}

PythonTest::~PythonTest()
{
    std::cout << "DETEL" << std::endl;
    //delete m_interpreter;
}

void PythonTest::run( const PythonTestData& data ) {
    msg_info("PythonTest") << "running " << data.filepath;
}

static bool ends_with(const std::string& suffix, const std::string& full){
    const std::size_t lf = full.length();
    const std::size_t ls = suffix.length();

    if(lf < ls) return false;

    return (0 == full.compare(lf - ls, ls, suffix));
}

static bool starts_with(const std::string& prefix, const std::string& full){
    const std::size_t lf = full.length();
    const std::size_t lp = prefix.length();

    if(lf < lp) return false;

    return (0 == full.compare(0, lp, prefix));
}

/// add a Python_test_data with given path
void PythonTestList::addTest( const std::string& filename,
                              const std::string& path,
                              const std::vector<std::string>& arguments)
{
    list.push_back( PythonTestData( filepath(path,filename), arguments ) );
}

void PythonTestList::addTestDir(const std::string& dir, const std::string& prefix) {

    std::vector<std::string> files;
    sofa::helper::system::FileSystem::listDirectory(dir, files);

    for(const std::string& file : files)
    {
        if( starts_with(prefix, file) && ends_with(".py", file) )
        {
            addTest(file, dir);
        }
    }
}

} // namespace sofapython3

