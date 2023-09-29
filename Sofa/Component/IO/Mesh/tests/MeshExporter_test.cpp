/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::execparams::defaultInstance;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::FileRepository;

#include <sofa/simulation/events/SimulationInitDoneEvent.h>
using sofa::simulation::SimulationInitDoneEvent;

#include <sofa/simulation/PropagateEventVisitor.h>
using sofa::simulation::PropagateEventVisitor;

#include <sofa/simulation/graph/SimpleApi.h>

using ::testing::Types;

namespace {
const std::string tempdir = FileRepository().getTempPath() ;

struct TestParam
{
    std::string extension;
    std::string format;
    sofa::type::vector<std::string> additionalExtensions; //Files that are exported in addition to the main file (example with the tetgen format)
};

class MeshExporter_test
        : public BaseSimulationTest,
          public ::testing::WithParamInterface<TestParam>
{
public:
    /// remove the file created...
    std::vector<string> dataPath;

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Grid");
    }

    void TearDown() override
    {
        for (const auto& pathToRemove : dataPath)
        {
            if (FileSystem::exists(pathToRemove))
            {
                if (FileSystem::isDirectory(pathToRemove))
                {
                    FileSystem::removeAll(pathToRemove);
                }
                else
                {
                    FileSystem::removeFile(pathToRemove);
                }
            }
        }
    }

    void checkBasicBehavior(const TestParam& params, const string& filename, const std::vector<string>& pathes)
    {
        dataPath = pathes;
        const string extension = params.extension;
        const string& format = params.format;

        EXPECT_MSG_NOEMIT(Error, Warning);
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <RegularGridTopology name='grid' n='6 6 6' min='-10 -10 -10' max='10 10 10' p0='-30 -10 -10' computeHexaList='1'/> \n"
                "   <MeshExporter name='exporter1' format='" << format << "' printLog='false' filename='" << filename << "' exportAtBegin='true' /> \n"
                "</Node>                                                           \n";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene1.str().c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        // SimulationInitDoneEvent is used to trigger exportAtBegin
        SimulationInitDoneEvent endInit;
        PropagateEventVisitor pe{sofa::core::execparams::defaultInstance(), &endInit};
        root->execute(pe);

        sofa::simulation::node::animate(root.get(), 0.5);

        for (auto& pathToCheck : pathes)
        {
            EXPECT_TRUE(FileSystem::exists(pathToCheck)) << "Problem with '" << pathToCheck << "'";
        }
    }


    void checkSimulationWriteEachNbStep(const TestParam& params, const string& filename, std::vector<string> pathes, unsigned int numstep)
    {
        dataPath = pathes;
        const string extension = params.extension;
        const string format = params.format;

        EXPECT_MSG_NOEMIT(Error, Warning);
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <RegularGridTopology name='grid' n='6 6 6' min='-10 -10 -10' max='10 10 10' p0='-30 -10 -10' computeHexaList='1'/> \n"
                "   <MeshExporter name='exporterA' format='" << format << "' printLog='false' filename='" << filename << "' exportEveryNumberOfSteps='5' /> \n"
                "</Node>                                                           \n";

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene1.str().c_str());

        ASSERT_NE(root.get(), nullptr);
        root->init(sofa::core::execparams::defaultInstance());

        for (unsigned int i = 0; i < numstep; i++)
        {
            sofa::simulation::node::animate(root.get(), 0.5);
        }

        for (auto& pathToCheck : pathes)
        {
            EXPECT_TRUE(FileSystem::exists(pathToCheck)) << "Problem with '" << pathToCheck << "'";
        }
    }
};

std::vector<TestParam> params {
    {"vtu", "vtkxml", {}},
    {"vtk", "vtk", {}},
    {"mesh", "netgen", {}},
    {"node", "tetgen", {"ele", "face"}},
    {"gmsh", "gmsh", {}}
};

/// run the tests
TEST_P(MeshExporter_test, checkBasicBehavior)
{
    const TestParam& params = GetParam();
    constexpr std::string_view filename = "outfile";
    std::vector<std::string> paths { std::string(filename) + "." + params.extension};
    for (const auto& addExtension : params.additionalExtensions)
    {
        paths.push_back(std::string(filename) + "." + addExtension);
    }
    ASSERT_NO_THROW(this->checkBasicBehavior(params, "outfile", paths));
}

TEST_P(MeshExporter_test, checkBasicBehaviorNoFileName)
{
    const TestParam& params = GetParam();
    constexpr std::string_view filename = "exporter1";
    std::vector<std::string> paths { std::string(filename) + "." + params.extension};
    for (const auto& addExtension : params.additionalExtensions)
    {
        paths.push_back(std::string(filename) + "." + addExtension);
    }
    ASSERT_NO_THROW(this->checkBasicBehavior(GetParam(), "", paths));
}

TEST_P(MeshExporter_test, checkBasicBehaviorInSubDirName)
{
    const TestParam& params = GetParam();
    ASSERT_NO_THROW(this->checkBasicBehavior(params, tempdir+"/outfile", {tempdir+"/outfile." + params.extension}));
}

TEST_P(MeshExporter_test, checkBasicBehaviorInInvalidSubDirName)
{
    const TestParam& params = GetParam();
    ASSERT_NO_THROW(this->checkBasicBehavior(params, tempdir+"/invalid/outfile", {tempdir+"/invalid"}));
}

TEST_P(MeshExporter_test, checkBasicBehaviorInInvalidLongSubDirName)
{
    const TestParam& params = GetParam();
    ASSERT_NO_THROW(this->checkBasicBehavior(params, tempdir+"/invalid1/invalid2/invalid3/outfile", {tempdir+"/invalid1/invalid2/invalid3"}));
}

TEST_P(MeshExporter_test, checkBasicBehaviorInInvalidRelativeDirName)
{
    const TestParam& params = GetParam();
    ASSERT_NO_THROW(this->checkBasicBehavior(params, "./invalidPath/outfile", {"./invalidPath"}));
}

TEST_P(MeshExporter_test, checkBasicBehaviorInValidDir)
{
    const TestParam& params = GetParam();
    constexpr std::string_view filename = "exporter1";
    std::vector<std::string> paths { tempdir + "/" + std::string(filename) + "." + params.extension};
    for (const auto& addExtension : params.additionalExtensions)
    {
        paths.push_back(tempdir + "/" + std::string(filename) + "." + addExtension);
    }
    ASSERT_NO_THROW(this->checkBasicBehavior(params, tempdir, paths));
}

TEST_P(MeshExporter_test, checkSimulationWriteEachNbStep)
{
    const TestParam& params = GetParam();
    constexpr std::string_view filename = "exporterA";
    std::vector<std::string> paths;
    constexpr unsigned int nbTimeSteps { 20 };
    constexpr unsigned int exportEveryNumberOfSteps { 5 };
    for (unsigned int i = 0; i < nbTimeSteps / exportEveryNumberOfSteps; ++i)
    {
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << (i+1);

        paths.push_back(FileSystem::append(tempdir, std::string(filename) + ss.str() + "." + params.extension));
        for (const auto& addExtension : params.additionalExtensions)
        {
            paths.push_back(FileSystem::append(tempdir, std::string(filename) + ss.str() + "." + addExtension));
        }
    }
    ASSERT_NO_THROW(this->checkSimulationWriteEachNbStep(params, tempdir, paths, nbTimeSteps)) ;
}

INSTANTIATE_TEST_SUITE_P(checkAllBehavior,
                         MeshExporter_test,
                         ::testing::ValuesIn(params));


}
