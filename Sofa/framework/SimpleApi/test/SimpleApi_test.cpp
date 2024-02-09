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
using sofa::testing::BaseSimulationTest ;

#include <sofa/simulation/graph/SimpleApi.h>
using namespace sofa ;
using namespace sofa::simpleapi ;

class SimpleApi_test : public BaseSimulationTest
{
public:
    bool testParamAPI();
    bool testParamString();
};

bool SimpleApi_test::testParamAPI()
{
    const Simulation::SPtr simu = createSimulation("DAG") ;
    const Node::SPtr root = createRootNode(simu, "root") ;

    const auto meca1 = createObject(root, "MechanicalObject", {
                                        {"name", "aMechanicalObject1"},
                                        {"position", "1 2 3"}
                                    });


    const auto meca2 = createObject(root, "MechanicalObject", {
                                        {"name", "aMechanicalObject2"},
                                        {"position", "1 2 3"}
                                    });

    EXPECT_EQ( (meca1->getName()), std::string("aMechanicalObject1") ) ;
    EXPECT_EQ( (meca2->getName()), std::string("aMechanicalObject2") ) ;

    return true ;
}

bool SimpleApi_test::testParamString()
{
    const Simulation::SPtr simu = createSimulation("DAG") ;
    const Node::SPtr root = createRootNode(simu, "root") ;

    const auto meca1 = createObject(root, "MechanicalObject", {
                                        {"name", "aMechanicalObject1"},
                                        {"position", "1 2 3"}
                                    });

    const auto meca2 = createObject(root, "MechanicalObject", {
                                        {"name", "aMechanicalObject2"},
                                        {"position", "1 2 3"}
                                    });

    EXPECT_EQ( (meca1->getName()), std::string("aMechanicalObject1") ) ;
    EXPECT_EQ( (meca2->getName()), std::string("aMechanicalObject2") ) ;

    return true;
}


TEST_F(SimpleApi_test, testParamAPI )
{
    ASSERT_TRUE( testParamAPI() );
}

TEST_F(SimpleApi_test, createParamString )
{
    ASSERT_TRUE( testParamString() );
}
