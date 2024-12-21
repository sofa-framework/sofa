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

#include <sofa/simpleapi/SimpleApi.h>
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

    sofa::simpleapi::importPlugin(Sofa.Component.StateContainer);

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

    simpleapi::importPlugin(Sofa.Component.StateContainer);

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

TEST(SimpleApi_test_solo, testIsSetWithDataLink)
{
    const Simulation::SPtr simu = createSimulation("DAG");
    const Node::SPtr root = createRootNode(simu, "root");

    // test not set
    const auto obj1 = createObject(root, "DefaultAnimationLoop", {
        {"name", "loop1"}
    });
    auto* objdata1 = obj1->findData("printLog");
    ASSERT_FALSE(objdata1->isSet());

    // test set
    const auto obj2 = createObject(root, "DefaultAnimationLoop", {
        {"name", "loop2"}, 
        {"printLog", "false"}
    });
    auto* objdata2 = obj2->findData("printLog");
    ASSERT_TRUE(objdata2->isSet());

    // test set through a link of a already created object
    const auto obj3 = createObject(root, "DefaultAnimationLoop", {
        {"name", "loop3"},
        {"printLog", "@/loop2.printLog"}
    });
    auto* objdata3 = obj3->findData("printLog");
    ASSERT_TRUE(objdata3->isSet());

    // test set through a link of a not yet created object (deferred linking)
    const auto obj4 = createObject(root, "DefaultAnimationLoop", {
        {"name", "loop4"},
        {"printLog", "@/loop5.printLog"}
    });
    const auto obj5 = createObject(root, "DefaultAnimationLoop", {
        {"name", "loop5"},
        {"printLog", "true"}
    });
    auto* objdata4 = obj4->findData("printLog");
    ASSERT_TRUE(objdata4->isSet());
    
    // test link with a wrong path (or non existent parent)
    const auto obj6 = createObject(root, "DefaultAnimationLoop", {
        {"name", "loop6"},
        {"printLog", "@/loop7.printLog"}
    });

    auto* objdata6 = obj6->findData("printLog");
    ASSERT_TRUE(objdata6->isSet());
    ASSERT_EQ(objdata6->getParent(), nullptr);

}
