#ifndef SOFAPYTHON3_PYTHONTEST_H
#define SOFAPYTHON3_PYTHONTEST_H

#include <string>
#include "config.h"
#include <sofa/helper/testing/BaseTest.h>

#include <boost/filesystem/path.hpp>
using boost::filesystem::path;

namespace sofapython3
{

using sofa::helper::testing::BaseTest;

/// a Python_test is defined by a python filepath and optional arguments
struct SOFAPYTHON3_API PythonTestData
{
    PythonTestData( const std::string& filepath, const std::string& testgroup, const std::vector<std::string>& arguments );
    std::string filepath;
    std::vector<std::string> arguments;
    std::string testgroup;
};

/// This function is used by gtest to print the content of the struct in a human friendly way
/// eg:
///        test.all_tests/2, where GetParam() = /path/to/file.py with args {1,2,3}
/// instead of the defautl googletest printer that output things like the following:
///        test.all_tests/2, where GetParam() = 56-byte object <10-48 EC-37 18-56 00-00 67-00-00-00>
void SOFAPYTHON3_API PrintTo(const PythonTestData& d, ::std::ostream* os);

/// utility to build a static list of Python_test_data
struct SOFAPYTHON3_API PythonTestList
{
    std::vector<PythonTestData> list;
protected:
    /// add a Python_test_data with given path
    void addTest(const std::string& filename,
                 const std::string& path="", const std::string &testgroup="",
                 const std::vector<std::string>& arguments=std::vector<std::string>(0) );

    /// add all the python test files in `dir` starting with `prefix`
    void addTestDir(const std::string& dir, const std::string& testgroup = "", const std::string& prefix = "" );

private:
    /// concatenate path and filename
    static std::string filepath( const std::string& path, const std::string& filename )
    {
        if( path!="" )
            return path+"/"+filename;
        else
            return filename;
    }
};

/// A test written in python (but not as a sofa class to perform unitary testing on python functions)
class SOFAPYTHON3_API PythonTest : public BaseTest,
        public ::testing::WithParamInterface<PythonTestData>
{
public:
    PythonTest();
    virtual ~PythonTest();

    void run( const PythonTestData& );

    /// This function is called by gtest to generate the test from the filename. This is nice
    /// As this allows to do mytest --gtest_filter=*MySomething*
    static std::string getTestName(const testing::TestParamInfo<PythonTestData>& p)
    {
        if(p.param.arguments.size()==0)
            return  std::to_string(p.index)+"_"+p.param.testgroup+path(p.param.filepath).stem().string();
        return  std::to_string(p.index)+"_"+p.param.testgroup+path(p.param.filepath).stem().string()
                                       +"_"+p.param.arguments[0];
    }
};

}

#endif /// PYTHONTEST_H_
