/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;

#include <sofa/helper/testing/BaseTest.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileSystem.h>

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager;

#include <SofaSimulationGraph/SimpleApi.h>
namespace simpleapi=sofa::simpleapi;

#include <sofa/helper/system/SetDirectory.h>
using sofa::helper::system::SetDirectory;

#include "PythonEnvironment.h"
#include "PythonTest.h"

MSG_REGISTER_CLASS(sofapython3::PythonTest, "SofaPython3::PythonTest")

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
PythonTestData::PythonTestData(const std::string& filepath, const std::string &testgroup, const std::vector<std::string>& arguments ) :
    filepath(filepath), arguments(arguments), testgroup{testgroup} {}


///////////////////////// PythonTest Definition  //////////////////////////////////////////////////
PythonTest::PythonTest()
{
}

PythonTest::~PythonTest()
{
}

void PythonTest::run( const PythonTestData& data )
{
    msg_info() << "running " << data.filepath;

    PythonEnvironment::Init();
    {
        EXPECT_MSG_NOEMIT(Error);
        PythonEnvironment::setArguments(data.filepath, data.arguments);
        simpleapi::importPlugin("SofaAllCommonComponents");
        sofa::simulation::setSimulation(simpleapi::createSimulation());

        try{
            PythonEnvironment::gil scoped_gil{__FUNCTION__};

            py::module::import("Sofa");
            py::object globals = py::module::import("__main__").attr("__dict__");
            py::module module;

            const char* filename = data.filepath.c_str();
            SetDirectory localDir(filename);
            std::string basename = SetDirectory::GetFileNameWithoutExtension(SetDirectory::GetFileName(filename).c_str());
            module = PythonEnvironment::importFromFile(basename, SetDirectory::GetFileName(filename),
                                                       globals);
            if(!module.attr("runTests"))
            {
                msg_error() << "Missing runTests function in file '"<< filename << "'";
                return ;
            }

            py::object runTests = module.attr("runTests");
            if( py::cast<bool>(runTests()) == false )
            {
                FAIL();
                return;
            }
        }catch(std::exception& e)
        {
            msg_error() << e.what();
        }
    }

    //PythonEnvironment::Release();
}

/// add a Python_test_data with given path
void PythonTestList::addTest( const std::string& filename,
                              const std::string& path,
                              const std::string& testgroup,
                              const std::vector<std::string>& arguments
                              )
{
    PythonEnvironment::Init();
    PythonEnvironment::gil scoped_gil{__FUNCTION__};

    py::module::import("Sofa");
    py::object globals = py::module::import("__main__").attr("__dict__");
    py::module module;

    const char* filenameC = filename.c_str();
    std::string fullpath = (path+"/"+filename);
    const char* pathC = fullpath.c_str();

    SetDirectory localDir(pathC);
    std::string basename = SetDirectory::GetFileNameWithoutExtension(SetDirectory::GetFileName(filenameC).c_str());
    module = PythonEnvironment::importFromFile(basename, SetDirectory::GetFileName(filenameC),
                                               globals);
    if(!module.attr("getTestsName"))
    {
        list.push_back( PythonTestData( filepath(path,filename), testgroup, arguments) );
        return ;
    }

    py::list names = module.attr("getTestsName")();

    for(auto& n : names)
    {
        std::vector<std::string> cargs;
        cargs.push_back(py::cast<std::string>(n));
        cargs.insert(cargs.end(), arguments.begin(), arguments.end());
        list.push_back( PythonTestData( filepath(path,filename), testgroup, cargs ) );
    }
}

void PythonTestList::addTestDir(const std::string& dir, const std::string& testgroup, const std::string& prefix)
{
    std::vector<std::string> files;
    sofa::helper::system::FileSystem::listDirectory(dir, files);

    for(const std::string& file : files)
    {
        if( sofa::helper::starts_with(prefix, file)
                && (sofa::helper::ends_with(".py", file) || sofa::helper::ends_with(".py3", file)))
        {
            addTest(file, dir, testgroup);
        }
    }
}

} /// namespace sofapython3
