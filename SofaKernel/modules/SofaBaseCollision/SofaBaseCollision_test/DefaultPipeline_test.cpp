/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA ::                                       *
*                                                                             *
* Contributors:                                                               *
*       damien.marchal@univ-lille1.fr                                         *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include<SofaBaseCollision/DefaultPipeline.h>
using sofa::component::collision::DefaultPipeline ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::ExecParams ;

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::MessageAsTestFailure ;
using sofa::helper::logging::ExpectMessage ;
using sofa::helper::logging::Message ;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace ;

namespace defaultpipeline_test
{

int initMessage(){
    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    return 0;
}

int messageInited = initMessage();

class TestDefaultPipeLine : public Sofa_test<double> {
public:
    void checkDefaultPipelineWithNoAttributes();
    void checkDefaultPipelineWithMissingIntersection();
    int checkDefaultPipelineWithMonkeyValueForDepth(int value);
};

void TestDefaultPipeLine::checkDefaultPipelineWithNoAttributes()
{
    MessageAsTestFailure warning(Message::Warning) ;
    MessageAsTestFailure error(Message::Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "  <DiscreteIntersection name='interaction'/>                                    \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}

void TestDefaultPipeLine::checkDefaultPipelineWithMissingIntersection()
{
    ExpectMessage warning(Message::Warning) ;
    MessageAsTestFailure error(Message::Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}

int TestDefaultPipeLine::checkDefaultPipelineWithMonkeyValueForDepth(int value)
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline' depth='"<< value <<"'/>                     \n"
             "  <DiscreteIntersection name='interaction'/>                                    \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    //ASSERT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    DefaultPipeline* clp = dynamic_cast<DefaultPipeline*>(root->getObject("pipeline")) ;
    //ASSERT_NE( clp, nullptr) ;

    int rv = clp->d_depth.getValue() ;
    clearSceneGraph();
    return rv;
}


TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithNoAttributes)
{
    this->checkDefaultPipelineWithNoAttributes();
}

TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithMissingIntersection)
{
    this->checkDefaultPipelineWithMissingIntersection();
}

TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithMonkeyValueForDepth_OpenIssue)
{
    std::vector<std::pair<int, bool>> testvalues = {
        std::make_pair(-1, false),
        std::make_pair( 0, true),
        std::make_pair( 2, true),
        std::make_pair(10, true),
        std::make_pair(1000, false)
    };

    for(auto is : testvalues){
        MessageAsTestFailure error(Message::Error) ;
        if(is.second){
            MessageAsTestFailure warning(Message::Warning) ;
            // Check the returned value.
            if(this->checkDefaultPipelineWithMonkeyValueForDepth(is.first)!=is.first){
                ADD_FAILURE() << "User provided depth parameter value '" << is.first << "' has been un-expectedly overriden." ;
            }
        }else{
            ExpectMessage warning(Message::Warning) ;
            // Check the default value.
            if(this->checkDefaultPipelineWithMonkeyValueForDepth(is.first)!=6){
                ADD_FAILURE() << "User provided invalid depth parameter value '" << is.first << "' have not been replaced with the default value = 6." ;
            }
        }
    }
}

} // defaultpipeline_test
