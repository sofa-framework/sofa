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

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/component/collision/detection/algorithm/CollisionPipeline.h>
using sofa::component::collision::detection::algorithm::CollisionPipeline ;

#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::graph::DAGSimulation ;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;
using sofa::core::ExecParams ;

#include <sofa/helper/BackTrace.h>
using sofa::helper::BackTrace;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/simulation/graph/SimpleApi.h>

namespace CollisionPipeline_test
{

int initMessage(){
    /// Install the backtrace so that we have more information in case of test segfault.
    BackTrace::autodump() ;
    return 0;
}

int messageInited = initMessage();

class TestCollisionPipeline : public BaseSimulationTest {
public:
    Node::SPtr root;

    void checkCollisionPipelineWithNoAttributes();
    void checkCollisionPipelineWithMissingIntersection();
    void checkCollisionPipelineWithMissingBroadPhase();
    void checkCollisionPipelineWithMissingNarrowPhase();
    void checkCollisionPipelineWithMissingContactManager();
    int checkCollisionPipelineWithMonkeyValueForDepth(int value);

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer"); 
        sofa::simpleapi::importPlugin("Sofa.Component.Collision");
    }

    void TearDown() override
    {
        if (root)
            sofa::simulation::node::unload(root);
    }
};

void TestCollisionPipeline::checkCollisionPipelineWithNoAttributes()
{
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <CollisionPipeline name='pipeline'/>                                         \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <CollisionResponse/>                                                         \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    root = SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str());
    
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;
}

void TestCollisionPipeline::checkCollisionPipelineWithMissingIntersection()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <CollisionPipeline name='pipeline'/>                                         \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <CollisionResponse/>                                                         \n"
             "</Node>                                                                        \n" ;

    root = SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;
}

void TestCollisionPipeline::checkCollisionPipelineWithMissingBroadPhase()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <CollisionPipeline name='pipeline'/>                                         \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <CollisionResponse/>                                                         \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    root = SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;
}
void TestCollisionPipeline::checkCollisionPipelineWithMissingNarrowPhase()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <CollisionPipeline name='pipeline'/>                                         \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <CollisionResponse/>                                                         \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    root = SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;
}
void TestCollisionPipeline::checkCollisionPipelineWithMissingContactManager()
{
    EXPECT_MSG_EMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <CollisionPipeline name='pipeline'/>                                           \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    root = SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str());
    ASSERT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* clp = root->getObject("pipeline") ;
    ASSERT_NE(clp, nullptr) ;

}

int TestCollisionPipeline::checkCollisionPipelineWithMonkeyValueForDepth(int dvalue)
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>                                                          \n"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <CollisionPipeline name='pipeline' depth='"<< dvalue <<"'/>                  \n"
             "  <BruteForceBroadPhase/>                                                      \n"
             "  <BVHNarrowPhase/>                                                            \n"
             "  <CollisionResponse/>                                                         \n"
             "  <DiscreteIntersection name='interaction'/>                                   \n"
             "</Node>                                                                        \n" ;

    root = SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str());
    //EXPECT_NE( (root.get()), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    const CollisionPipeline* clp = dynamic_cast<CollisionPipeline*>(root->getObject("pipeline")) ;
    //ASSERT_NE( (clp), nullptr) ;

    const int rv = clp->d_depth.getValue() ;

    return rv;
}


TEST_F(TestCollisionPipeline, checkCollisionPipelineWithNoAttributes)
{
    this->checkCollisionPipelineWithNoAttributes();
}

TEST_F(TestCollisionPipeline, checkCollisionPipelineWithMissingIntersection)
{
    this->checkCollisionPipelineWithMissingIntersection();
}

TEST_F(TestCollisionPipeline, checkCollisionPipelineWithMissingBroadPhase)
{
    this->checkCollisionPipelineWithMissingBroadPhase();
}

TEST_F(TestCollisionPipeline, checkCollisionPipelineWithMissingNarrowPhase)
{
    this->checkCollisionPipelineWithMissingNarrowPhase();
}

TEST_F(TestCollisionPipeline, checkCollisionPipelineWithMissingContactManager)
{
    this->checkCollisionPipelineWithMissingContactManager();
}

TEST_F(TestCollisionPipeline, checkCollisionPipelineWithMonkeyValueForDepth_OpenIssue)
{
    const std::vector<std::pair<int, bool>> testvalues = {
        {-1, false},
        { 0, true},
        { 2, true},
        {10, true},
        {1000, true}
    };

    for(const auto& [depthValue, validValue] : testvalues)
    {
        EXPECT_MSG_NOEMIT(Error) ;

        if (validValue)
        {
            EXPECT_MSG_NOEMIT(Warning) ;

            // Check the returned value.
            if(this->checkCollisionPipelineWithMonkeyValueForDepth(depthValue) != depthValue)
            {
                ADD_FAILURE() << "User provided depth parameter value '" << depthValue << "' has been un-expectedly overriden." ;
            }
        }
        else
        {
            EXPECT_MSG_EMIT(Warning) ;

            // Check the default value.
            if(this->checkCollisionPipelineWithMonkeyValueForDepth(depthValue) != CollisionPipeline::defaultDepthValue)
            {
                ADD_FAILURE() << "User provided invalid depth parameter value '" << depthValue << "' and has not been replaced with the default value = " << CollisionPipeline::defaultDepthValue;
            }
        }
    }
}

} // CollisionPipeline_test
