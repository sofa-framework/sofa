/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
/******************************************************************************
 * Contributors:                                                              *
 *    - damien.marchal@univ-lille1.fr                                         *
 *****************************************************************************/
#include <vector>
using std::vector;

#include <string>
using std::string;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::ExecParams ;

#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem ;

#include <SofaTest/Sofa_test.h>

class OBJExporter_test : public sofa::Sofa_test<>{
public:
    /// remove the file created...
    std::vector<std::string> dataPath ;

    void TearDown()
    {
        for(auto& pathToRemove : dataPath)
        {
            if(FileSystem::exists(pathToRemove))
               FileSystem::removeAll(pathToRemove) ;
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
                "   <OBJExporter name='exporter1' printLog='true' filename='"<< filename << "' exportAtBegin='true' /> \n"
                "</Node>                                                           \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene1.str().c_str(),
                                                          scene1.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        sofa::simulation::getSimulation()->animate(root.get(), 0.5);

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
                "   <OBJExporter name='exporterA' printLog='true' filename='"<< filename << "' exportEveryNumberOfSteps='5' /> \n"
                "</Node>                                                           \n" ;

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                          scene1.str().c_str(),
                                                          scene1.str().size()) ;

        ASSERT_NE(root.get(), nullptr) ;
        root->init(ExecParams::defaultInstance()) ;

        for(unsigned int i=0;i<numstep;i++)
        {
            sofa::simulation::getSimulation()->animate(root.get(), 0.5);
        }

        for(auto& pathToCheck : pathes)
        {
            EXPECT_TRUE( FileSystem::exists(pathToCheck) ) << "Problem with '" << pathToCheck  << "'";
        }
    }
};

/// run the tests
TEST_F( OBJExporter_test, checkBasicBehavior) {
    ASSERT_NO_THROW( this->checkBasicBehavior("outfile", {"outfile.obj","outfile.mtl"}) ) ;
}

TEST_F( OBJExporter_test, checkBasicBehaviorNoFileName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("", {"exporter1.obj","exporter1.mtl"}) ) ;
}

TEST_F( OBJExporter_test, checkBasicBehaviorInSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("/tmp/outfile", {"/tmp/outfile.obj","/tmp/outfile.mtl"}) ) ;
}

TEST_F( OBJExporter_test, checkBasicBehaviorInInvalidSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("/tmp/invalid/outfile", {"/tmp/invalid"}) ) ;
}

TEST_F( OBJExporter_test, checkBasicBehaviorInInvalidLongSubDirName) {
    ASSERT_NO_THROW( this->checkBasicBehavior("/tmp/invalid1/invalid2/invalid3/outfile", {"/tmp/invalid1"})) ;
}

TEST_F( OBJExporter_test, checkBasicBehaviorInInvalidRelativeDirName) {
   ASSERT_NO_THROW( this->checkBasicBehavior("./invalidPath/outfile", {"./invalidPath"}) ) ;
}

TEST_F( OBJExporter_test, checkBasicBehaviorInValidDir) {
   this->checkBasicBehavior("/tmp", {"/tmp/exporter1.obj", "/tmp/exporter1.mtl"})  ;
}

TEST_F( OBJExporter_test, checkSimulationWriteEachNbStep) {
   this->checkSimulationWriteEachNbStep("/tmp", {"/tmp/exporterA00001.obj", "/tmp/exporterA00001.mtl",
                                                 "/tmp/exporterA00002.obj", "/tmp/exporterA00002.mtl",
                                                 "/tmp/exporterA00003.obj", "/tmp/exporterA00003.mtl",
                                                 "/tmp/exporterA00004.obj", "/tmp/exporterA00004.mtl"}, 20)  ;
}
