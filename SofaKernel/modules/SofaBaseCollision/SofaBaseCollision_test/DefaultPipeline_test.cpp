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

#include <sofa/core/fwd.h>
using sofa::core::execparams::defaultInstance;

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

class TestDefaultPipeLine : public Sofa_test<> {
public:
    void checkDefaultPipelineWithNoAttributes();
    void checkDefaultPipelineWithMissingIntersection();
    void checkDefaultPipelineWithMissingBroadPhase();
    void checkDefaultPipelineWithMissingNarrowPhase();
    void checkDefaultPipelineWithMissingContactManager();
    int checkDefaultPipelineWithMonkeyValueForDepth(int value);
};

void TestDefaultPipeLine::checkDefaultPipelineWithNoAttributes()
{
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <DefaultContactManager/>                                                     \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}

void TestDefaultPipeLine::checkDefaultPipelineWithMissingIntersection()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <DefaultContactManager/>                                                     \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}

void TestDefaultPipeLine::checkDefaultPipelineWithMissingBroadPhase()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <DefaultContactManager/>                                                     \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}
void TestDefaultPipeLine::checkDefaultPipelineWithMissingNarrowPhase()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <DefaultContactManager/>                                                     \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}
void TestDefaultPipeLine::checkDefaultPipelineWithMissingContactManager()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline'/>                                           \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

    clearSceneGraph();
}

int TestDefaultPipeLine::checkDefaultPipelineWithMonkeyValueForDepth(int dvalue)
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <DefaultPipeline name='pipeline' depth='"<< dvalue <<"'/>                    \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <DefaultContactManager/>                                                     \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    //EXPECT_NE( (root.get()), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    DefaultPipeline* clp = dynamic_cast<DefaultPipeline*>(root->getObject("pipeline")) ;
    //ASSERT_NE( (clp), nullptr) ;

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

TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithMissingBroadPhase)
{
    this->checkDefaultPipelineWithMissingBroadPhase();
}

TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithMissingNarrowPhase)
{
    this->checkDefaultPipelineWithMissingNarrowPhase();
}

TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithMissingContactManager)
{
    this->checkDefaultPipelineWithMissingContactManager();
}

TEST_F(TestDefaultPipeLine, checkDefaultPipelineWithMonkeyValueForDepth_OpenIssue)
{
    const std::vector<std::pair<int, bool>> testvalues = {
        {-1, false},
        { 0, true},
        { 2, true},
        {10, true},
        {1000, true}
    };

    for(const auto& is : testvalues)
    {
        EXPECT_MSG_NOEMIT(Error) ;

        if (is.second)
        {
            EXPECT_MSG_NOEMIT(Warning) ;

            // Check the returned value.
            if(this->checkDefaultPipelineWithMonkeyValueForDepth(is.first) != is.first)
            {
                ADD_FAILURE() << "User provided depth parameter value '" << is.first << "' has been un-expectedly overriden." ;
            }
        }
        else
        {
            EXPECT_MSG_EMIT(Warning) ;

            // Check the default value.
            if(this->checkDefaultPipelineWithMonkeyValueForDepth(is.first) != DefaultPipeline::defaultDepthValue)
            {
                ADD_FAILURE() << "User provided invalid depth parameter value '" << is.first << "' and has not been replaced with the default value = " << DefaultPipeline::defaultDepthValue;
            }
        }
    }
}

} // defaultpipeline_test
