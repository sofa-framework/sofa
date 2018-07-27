#ifndef SOFA_STANDARDTEST_Python_test_H
#define SOFA_STANDARDTEST_Python_test_H

#include <gtest/gtest.h>
#include <string>
#include <SofaPython/SceneLoaderPY.h>

#include "InitPlugin_test.h"

namespace sofa {


/// a Python_test is defined by a python filepath and optional arguments
struct SOFA_SOFATEST_API Python_test_data
{
    Python_test_data( const std::string& filepath,
                      const std::vector<std::string>& arguments )
        : filepath(filepath), arguments(arguments) {}

    std::string filepath;
    std::vector<std::string> arguments; // argc/argv in the python script
};

/// This function is used by gtest to print the content of the struct in a human friendly way
/// eg:
///        test.all_tests/2, where GetParam() = /path/to/file.py with args {1,2,3}
/// instead of the defautl googletest printer that output things like the following:
///        test.all_tests/2, where GetParam() = 56-byte object <10-48 EC-37 18-56 00-00 67-00-00-00>
void SOFA_SOFATEST_API PrintTo(const sofa::Python_test_data& d, ::std::ostream* os);

/// utility to build a static list of Python_test_data
struct SOFA_SOFATEST_API Python_test_list
{
    std::vector<Python_test_data> list;
protected:
    /// add a Python_test_data with given path
    void addTest( const std::string& filename,
                  const std::string& path="",
                  const std::vector<std::string>& arguments=std::vector<std::string>(0) )
    {
        list.push_back( Python_test_data( filepath(path,filename), arguments ) );
    }


    /// add all the python test files in `dir` starting with `prefix`
    void addTestDir(const std::string& dir, const std::string& prefix = "test_");



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
class SOFA_SOFATEST_API Python_test : public ::testing::TestWithParam<Python_test_data> {

protected:

    simulation::SceneLoaderPY loader;

public:

#ifdef WIN32 // Fix for linking in Visual Studio (the functions are not exported by the gtest library)
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
#endif

    struct result {
        result(bool value) : value( value ) { }
        bool value;
    };

    void run( const Python_test_data& );

    Python_test();

};


/// A test written as a sofa scene in python
class SOFA_SOFATEST_API Python_scene_test : public Python_test {

public:
    std::size_t max_steps;

    Python_scene_test();
    void run( const Python_test_data& );

};


}

#endif
