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
using std::vector;
using testing::Types;

#include <boost/filesystem.hpp>
namespace {
std::string tempdir = boost::filesystem::temp_directory_path().string() ;

class MeshExporter_test : public sofa::Sofa_test<>,
                          public ::testing::WithParamInterface<vector<string>>
{
public:
    /// remove the file created...
    std::vector<string> dataPath ;

    void TearDown()
    {
        for(auto& pathToRemove : dataPath)
        {
            if(FileSystem::exists(pathToRemove))
               FileSystem::removeAll(pathToRemove) ;
       }
    }

    void checkBasicBehavior(const std::vector<string>& params, const string& filename, std::vector<string> pathes){
        dataPath = pathes ;
        const string extension = params[0] ;
        const string format = params[1] ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <RegularGridTopology name='grid' n='6 6 6' min='-10 -10 -10' max='10 10 10' p0='-30 -10 -10' computeHexaList='0'/> \n"
                "   <MeshExporter name='exporter1' format='"<< format <<"' printLog='true' filename='"<< filename << "' exportAtBegin='true' /> \n"
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


    void checkSimulationWriteEachNbStep(const std::vector<string>& params, const string& filename, std::vector<string> pathes, unsigned int numstep){
        dataPath = pathes ;
        const string extension = params[0] ;
        const string format = params[1] ;

        EXPECT_MSG_NOEMIT(Error, Warning) ;
        std::stringstream scene1;
        scene1 <<
                "<?xml version='1.0'?> \n"
                "<Node 	name='Root' gravity='0 0 0' time='0' animate='0'   >       \n"
                "   <DefaultAnimationLoop/>                                        \n"
                "   <MechanicalObject position='0 1 2 3 4 5 6 7 8 9'/>             \n"
                "   <RegularGridTopology name='grid' n='6 6 6' min='-10 -10 -10' max='10 10 10' p0='-30 -10 -10' computeHexaList='0'/> \n"
                "   <MeshExporter name='exporterA' format='"<< format <<"' printLog='true' filename='"<< filename << "' exportEveryNumberOfSteps='5' /> \n"
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


std::vector<std::vector<string>> params={
    {"vtu", "vtkxml"},
    {"vtk", "vtk"},
    {"mesh", "netgen"},
    {"node", "tetgen"},
    {"gmsh", "gmsh"}
};

#define NUM_PARAMS (unsigned int)2

/// run the tests
TEST_P( MeshExporter_test, checkBasicBehavior) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW( this->checkBasicBehavior(params, "outfile", {"outfile."+params[0]}) ) ;
}

TEST_P( MeshExporter_test, checkBasicBehaviorNoFileName) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW( this->checkBasicBehavior(GetParam(), "", {"exporter1."+params[0]}) ) ;
}

TEST_P( MeshExporter_test, checkBasicBehaviorInSubDirName) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW( this->checkBasicBehavior(params, tempdir+"/outfile", {tempdir+"/outfile."+params[0]}) ) ;
}

TEST_P( MeshExporter_test, checkBasicBehaviorInInvalidSubDirName) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW( this->checkBasicBehavior(params, tempdir+"/invalid/outfile", {tempdir+"/invalid"}) ) ;
}

TEST_P( MeshExporter_test, checkBasicBehaviorInInvalidLongSubDirName) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW( this->checkBasicBehavior(params, tempdir+"/invalid1/invalid2/invalid3/outfile", {tempdir+"/invalid1/invalid2/invalid3"})) ;
}

TEST_P( MeshExporter_test, checkBasicBehaviorInInvalidRelativeDirName) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW( this->checkBasicBehavior(params, "./invalidPath/outfile", {"./invalidPath"}) ) ;
}

TEST_P( MeshExporter_test, checkBasicBehaviorInValidDir) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW(this->checkBasicBehavior(params, tempdir, {tempdir+"/exporter1."+params[0]}))  ;
}

TEST_P( MeshExporter_test, checkSimulationWriteEachNbStep) {
    std::vector<string> params = GetParam() ;
    ASSERT_EQ(params.size(), NUM_PARAMS );
    ASSERT_NO_THROW(this->checkSimulationWriteEachNbStep(params, tempdir, {tempdir+"/exporterA00001."+params[0],
                                                        tempdir+"/exporterA00002."+params[0],
                                                        tempdir+"/exporterA00003."+params[0],
                                                        tempdir+"/exporterA00004."+params[0]}, 20)) ;
}

INSTANTIATE_TEST_CASE_P(checkAllBehavior,
                        MeshExporter_test,
                        ::testing::ValuesIn(params));


}
