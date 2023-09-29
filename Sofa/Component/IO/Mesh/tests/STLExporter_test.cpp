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

#include <sofa/simulation/graph/SimpleApi.h>

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


namespace{
const std::string tempdir = FileRepository().getTempPath() ;


class STLExporter_test : public BaseSimulationTest {
public:
    /// remove the file created...
    std::vector<std::string> dataPath ;

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Visual");
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

    void checkBasicBehavior(const std::string& filename, std::vector<std::string> pathes){
        dataPath = pathes ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <RequiredPlugin name='Sofa.Component.IO.Mesh' />               \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <MeshOBJLoader name='loader' filename='mesh/liver-smooth.obj'/> \n"
                "   <VisualModel src='@loader'/>                                      \n"
                "   <STLExporter name='exporter1' printLog='false' filename='"<< filename << "' exportAtBegin='true' /> \n"
                "</Node>                                                           \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene1.str().c_str());

        ASSERT_NE(root.get(), nullptr) << scene1.str() ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        // SimulationInitDoneEvent is used to trigger exportAtBegin
        SimulationInitDoneEvent endInit;
        PropagateEventVisitor pe{sofa::core::execparams::defaultInstance(), &endInit};
        root->execute(pe);

        sofa::simulation::node::animate(root.get(), 0.5);

        for(auto& pathToCheck : pathes)
        {
            EXPECT_TRUE( FileSystem::exists(pathToCheck) ) << "Problem with '" << pathToCheck  << "'"<< std::endl
                                                           << "================= scene dump ==========================="
                                                           << scene1.str() ;
        }
    }


    void checkSimulationWriteEachNbStep(const std::string& filename, std::vector<std::string> pathes, unsigned int numstep){
        dataPath = pathes ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <MeshOBJLoader name='loader' filename='mesh/liver-smooth.obj'/> \n"
                "   <VisualModel src='@loader'/>                                      \n"
                "   <STLExporter name='exporterA' printLog='false' filename='"<< filename << "' exportEveryNumberOfSteps='5' /> \n"
                "</Node>                                                           \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene1.str().c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        for(unsigned int i=0;i<numstep;i++)
        {
            sofa::simulation::node::animate(root.get(), 0.5);
        }

        for(auto& pathToCheck : pathes)
        {
            EXPECT_TRUE( FileSystem::exists(pathToCheck) ) << "Problem with '" << pathToCheck  << "'";
        }
    }
};

/// run the tests
TEST_F( STLExporter_test, checkBasicBehavior) {
    ASSERT_NO_THROW( this->checkBasicBehavior("outfile", {"outfile.stl"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorNoFileName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("", {"exporter1.stl"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/outfile", {tempdir+"/outfile.stl"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInInvalidSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/invalid/outfile", {tempdir+"/invalid"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInInvalidLongSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/invalid1/invalid2/invalid3/outfile", {tempdir+"/invalid1/invalid2/invalid3"})) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInInvalidRelativeDirName) {
   ASSERT_NO_THROW( this->checkBasicBehavior("./invalidPath/outfile", {"./invalidPath"}) ) ;
}

TEST_F( STLExporter_test, checkBasicBehaviorInValidDir) {
   this->checkBasicBehavior(tempdir, {tempdir+"/exporter1.stl"})  ;
}

TEST_F( STLExporter_test, checkSimulationWriteEachNbStep) {
   this->checkSimulationWriteEachNbStep(tempdir, {tempdir+"/exporterA00001.stl",
                                                 tempdir+"/exporterA00002.stl",
                                                 tempdir+"/exporterA00003.stl",
                                                 tempdir+"/exporterA00004.stl"}, 20)  ;
}
}
