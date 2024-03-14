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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/simulation/Node.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

using sofa::simulation::SceneLoaderXML ;
using sofa::simulation::Node ;

using sofa::core::execparams::defaultInstance; 

namespace sofa
{

struct RequiredPlugin_test : public BaseSimulationTest
{
    void testNotExistingPlugin()
    {
        EXPECT_MSG_EMIT(Error);

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <RequiredPlugin name=\"notExist\" pluginName=\"SofaNotExist\" />            \n"
                 "</Node>                                                                        \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;
    }

    void testNoParameter()
    {
        EXPECT_MSG_EMIT(Error) ;

        std::stringstream scene ;
        scene << "<?xml version='1.0'?>"
                 "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
                 "   <RequiredPlugin />            \n"
                 "</Node>                                                                        \n" ;

        const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());

        ASSERT_NE(root.get(), nullptr) ;
        root->init(sofa::core::execparams::defaultInstance()) ;
    }
};

TEST_F(RequiredPlugin_test, testNotExistingPlugin ) { testNotExistingPlugin(); }
TEST_F(RequiredPlugin_test, testNoParameter ) { testNoParameter(); }

}
