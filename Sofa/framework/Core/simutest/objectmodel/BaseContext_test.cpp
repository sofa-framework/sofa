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
#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseContext ;

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

#include <sofa/component/sceneutility/InfoComponent.h>
using sofa::component::sceneutility::InfoComponent;

class BaseContext_test: public BaseSimulationTest
{
public:
    void testGetObjects()
    {
        EXPECT_MSG_NOEMIT(Error, Warning) ;
        const std::string scene = R"(
            <?xml version='1.0'?>
            <Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >
               <RequiredPlugin name='Sofa.Component.SceneUtility'/>
               <DefaultAnimationLoop />
               <DefaultVisualManagerLoop />
               <InfoComponent/>
               <Node name='child1'>
                  <InfoComponent/>
                  <InfoComponent/>
                  <MechanicalObject />
                  <Node name='child2'>
                      <InfoComponent/>
                      <InfoComponent/>
                  </Node>
               </Node>
            </Node>
        )";

        SceneInstance c("xml", scene) ;
        c.initScene() ;

        Node* root = c.root.get() ;
        ASSERT_NE(root, nullptr) ;
        BaseContext* context = root->getChild("child1")->getContext() ;

        /// Query a specific model in a container, this is the old API
        std::vector<InfoComponent*> results ;
        context->getObjects<InfoComponent, std::vector<InfoComponent*> >( &results ) ;
        ASSERT_EQ( results.size() , 3 ) ;

        /// Query a specific model with a nicer syntax
        std::vector<InfoComponent*> results2 ;
        ASSERT_EQ( context->getObjects(results2).size(), 3 ) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        /// getObjects()'s default search direction is SearchUp
        /// so it will get the current objects in the context + the ones upper
        /// (here 3 in child1 + 3 in root itself + 1 required plugin)
        /// So it will find 6 base objects in the scene.
        for(const auto& m : context->getObjects() ) { SOFA_UNUSED(m); }
        ASSERT_EQ( context->getObjects().size(), 7 ) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        for(const auto& m : context->getObjects(BaseContext::SearchDirection::SearchDown) ) { SOFA_UNUSED(m); }
        ASSERT_EQ( context->getObjects(BaseContext::SearchDirection::SearchDown).size(), 5) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        ASSERT_EQ( context->getObjects<InfoComponent>(BaseContext::SearchDirection::SearchDown).size(), 4) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        ASSERT_EQ( context->getObjects(BaseContext::SearchDirection::Local).size(), 3) ;
        ASSERT_EQ( context->getObjects<InfoComponent>(BaseContext::SearchDirection::Local).size(), 2) ;
    }
};

TEST_F(BaseContext_test , testGetObjects )
{
    this->testGetObjects();
}

