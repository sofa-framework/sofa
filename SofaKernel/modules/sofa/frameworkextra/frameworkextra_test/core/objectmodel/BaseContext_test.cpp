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
 * Contributors:
 *     - damien.marchal@univ-lille1.fr
 *****************************************************************************/

#include <sofa/core/objectmodel/BaseContext.h>
using sofa::core::objectmodel::BaseContext ;

#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;
using sofa::simulation::Node ;

using sofa::core::visual::VisualModel ;

class BaseContext_test: public BaseSimulationTest
{
public:
    void testGetObjects()
    {
        EXPECT_MSG_NOEMIT(Error, Warning) ;
        importPlugin("SofaAllCommonComponents") ;
        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <OglModel/>                                                             \n"
                 "   <Node name='child1'>                                                    \n"
                 "      <OglModel/>                                                          \n"
                 "      <OglModel/>                                                          \n"
                 "      <MechanicalObject />                                                 \n"
                 "      <Node name='child2'>                                                 \n"
                 "          <OglModel/>                                                      \n"
                 "          <OglModel/>                                                      \n"
                 "      </Node>                                                              \n"
                 "   </Node>                                                                 \n"
                 "</Node>                                                                    \n" ;

        SceneInstance c("xml", scene.str()) ;
        c.initScene() ;

        Node* root = c.root.get() ;
        ASSERT_NE(root, nullptr) ;
        BaseContext* context = root->getChild("child1")->getContext() ;

        /// Query a specific model in a container, this is the old API
        std::vector<VisualModel*> results ;
        context->getObjects<VisualModel, std::vector<VisualModel*> >( &results ) ;
        ASSERT_EQ( results.size() , 3 ) ;

        /// Query a specific model with a nicer syntax
        std::vector<VisualModel*> results2 ;
        ASSERT_EQ( context->getObjects(results2).size(), 3 ) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        /// So there is 4 base object in the scene.
        for(auto& m : context->getObjects() ) { SOFA_UNUSED(m); }
        ASSERT_EQ( context->getObjects().size(), 4 ) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        for(auto& m : context->getObjects(BaseContext::SearchDirection::SearchDown) ) { SOFA_UNUSED(m); }
        ASSERT_EQ( context->getObjects(BaseContext::SearchDirection::SearchDown).size(), 5) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        ASSERT_EQ( context->getObjects<VisualModel>(BaseContext::SearchDirection::SearchDown).size(), 4) ;

        /// Query a specific model with a compact syntax, this returns std::vector<BaseObject*>
        ASSERT_EQ( context->getObjects(BaseContext::SearchDirection::Local).size(), 3) ;
        ASSERT_EQ( context->getObjects<VisualModel>(BaseContext::SearchDirection::Local).size(), 2) ;
    }
};

TEST_F(BaseContext_test , testGetObjects )
{
    this->testGetObjects();
}

