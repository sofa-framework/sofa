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

namespace{
const std::string tempdir = FileRepository().getTempPath() ;


class VisualModelOBJExporter_test : public BaseSimulationTest {
public:
    /// remove the file created...
    std::vector<std::string> dataPath ;

    void doSetUp() override
    {
    }

    void doTearDown() override
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
                "   <VisualModelOBJExporter name='exporter1' printLog='false' filename='"<< filename << "' exportAtBegin='true' /> \n"
                "</Node>                                                           \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;

        // SimulationInitDoneEvent is used to trigger exportAtBegin
        SimulationInitDoneEvent endInit;
        PropagateEventVisitor pe{sofa::core::execparams::defaultInstance(), &endInit};
        root->execute(pe);

        sofa::simulation::node::animate(root.get(), 0.5);

        for(auto& pathToCheck : pathes)
        {
            EXPECT_TRUE( FileSystem::exists(pathToCheck) ) << "Problem with '" << pathToCheck  << "'";
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
                "   <VisualModelOBJExporter name='exporterA' printLog='false' filename='"<< filename << "' exportEveryNumberOfSteps='5' /> \n"
                "</Node>                                                           \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str());

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
TEST_F( VisualModelOBJExporter_test, checkBasicBehavior) {
    ASSERT_NO_THROW( this->checkBasicBehavior("outfile", {"outfile.obj","outfile.mtl"}) ) ;
}

TEST_F( VisualModelOBJExporter_test, checkBasicBehaviorNoFileName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("", {"exporter1.obj","exporter1.mtl"}) ) ;
}

TEST_F( VisualModelOBJExporter_test, checkBasicBehaviorInSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/outfile", {tempdir+"/outfile.obj",tempdir+"/outfile.mtl"}) ) ;
}

TEST_F( VisualModelOBJExporter_test, checkBasicBehaviorInInvalidSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/invalid/outfile", {tempdir+"/invalid"}) ) ;
}

TEST_F( VisualModelOBJExporter_test, checkBasicBehaviorInInvalidLongSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior(tempdir+"/invalid1/invalid2/invalid3/outfile", {tempdir+"/invalid1"})) ;
}

TEST_F( VisualModelOBJExporter_test, checkBasicBehaviorInInvalidRelativeDirName) {
   ASSERT_NO_THROW( this->checkBasicBehavior("./invalidPath/outfile", {"./invalidPath"}) ) ;
}

TEST_F( VisualModelOBJExporter_test, checkBasicBehaviorInValidDir) {
   this->checkBasicBehavior(tempdir, {tempdir+"/exporter1.obj", tempdir+"/exporter1.mtl"})  ;
}

TEST_F( VisualModelOBJExporter_test, checkSimulationWriteEachNbStep) {
   this->checkSimulationWriteEachNbStep(tempdir, {tempdir+"/exporterA00001.obj", tempdir+"/exporterA00001.mtl",
                                                 tempdir+"/exporterA00002.obj", tempdir+"/exporterA00002.mtl",
                                                 tempdir+"/exporterA00003.obj", tempdir+"/exporterA00003.mtl",
                                                 tempdir+"/exporterA00004.obj", tempdir+"/exporterA00004.mtl"}, 20)  ;
}
}
