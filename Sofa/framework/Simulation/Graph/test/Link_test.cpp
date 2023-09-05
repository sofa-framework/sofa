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
using namespace sofa::simpleapi ;

#include "Node_test.h"

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;
#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/objectmodel/Link.h>
using sofa::core::objectmodel::SingleLink ;
using sofa::core::objectmodel::BaseLink ;

namespace sofa {

// tests Link features that are dependent on a graph structure
struct Link_test : public BaseSimulationTest
{
    void setLinkedBase_test()
    {
        SceneInstance si("root") ;

        auto aBaseObject = sofa::core::objectmodel::New<BaseObject>();
        sofa::core::objectmodel::Base* aBasePtr = aBaseObject.get();
        si.root->addObject(aBaseObject);

        using sofa::core::objectmodel::BaseNode;
        BaseLink::InitLink<BaseObject> initObjectLink(aBaseObject.get(), "objectlink", "");
        BaseLink::InitLink<BaseObject> initNodeLink(aBaseObject.get(), "nodelink", "");
        SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > objectLink(initObjectLink) ;
        SingleLink<BaseObject, BaseNode, BaseLink::FLAG_NONE > nodeLink(initNodeLink);

        // objectLink.add(aBasePtr); //< not possible because of template type specification

        objectLink.setLinkedBase(aBaseObject.get());
        EXPECT_EQ(objectLink.getLinkedBase(), aBaseObject.get());

        // EXPECT_MSG_EMIT(Error);
        nodeLink.setLinkedBase(aBasePtr); //< should emit error because BaseNode template type is incompatible with aBasePtr which is a BaseObject. But read() isn't implemented that way...

        EXPECT_NE(nodeLink.getLinkedBase(), aBasePtr);
    }

    void read_multilink_test()
    {
        const SceneInstance si("root") ;
        const BaseObject::SPtr A = sofa::core::objectmodel::New<BaseObject>();
        const BaseObject::SPtr B = sofa::core::objectmodel::New<BaseObject>();
        const BaseObject::SPtr C = sofa::core::objectmodel::New<BaseObject>();
        si.root->addObject(A);
        si.root->addObject(B);

        const BaseLink::InitLink<BaseObject> il1(B.get(), "l1", "");
        MultiLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > withOwner(il1) ;

        // 1. test with valid link & owner
        EXPECT_TRUE(withOwner.read("@/B"));

        // 2. setting C's context
        si.root->addObject(C);

        EXPECT_TRUE(withOwner.read("@/C"));
        EXPECT_TRUE(withOwner.read("@/B @/C"));
    }

    void read_test()
    {
        SceneInstance si("root") ;
        BaseObject::SPtr A = sofa::core::objectmodel::New<BaseObject>();
        BaseObject::SPtr B = sofa::core::objectmodel::New<BaseObject>();
        si.root->addObject(A);
        BaseLink::InitLink<BaseObject> il1(B.get(), "l1", "");
        SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > withOwner(il1) ;
        SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > withoutOwner;
        withoutOwner.setOwner(nullptr);

        // 1. test with invalid link & no owner
        EXPECT_FALSE(withoutOwner.read("@/B")); // should return false as link has no owner

        // 2. test with valid link but no owner
        EXPECT_FALSE(withoutOwner.read("@"+A->getPathName())); // should return false as we have no owner to call findLinkDest with

        // 3. test with valid link & valid owner but no context
        EXPECT_TRUE(withOwner.read("@"+A->getPathName())); // should return true as the owner could be added later in the graph

        // setting B's context
        si.root->addObject(B);

        // 4. test with invalid link but valid owner
        {
            EXPECT_MSG_EMIT(Error);
            EXPECT_FALSE(withOwner.read("/A")) << "should return false as the link is invalid (should start with '@')";
        }
        EXPECT_TRUE(withOwner.read("@/plop")); // same as 3: plop could be added later in the graph, after init()

        // test with valid link & valid owner
        EXPECT_TRUE(withOwner.read("@/A")); // standard call: everything is initialized, link is OK, owner exists and has a context
    }

    // introduced initially in https://github.com/sofa-framework/sofa/pull/1436
    void read_test_tofix()
    {
        const SceneInstance si("root");
        const BaseObject::SPtr A = sofa::core::objectmodel::New<BaseObject>();
        const BaseObject::SPtr B = sofa::core::objectmodel::New<BaseObject>();
        si.root->addObject(A);
        const BaseLink::InitLink<BaseObject> il1(B.get(), "l1", "");
        SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > withOwner(il1);
        SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > withoutOwner;
        withoutOwner.setOwner(nullptr);

        // Here link is OK, but points to a BaseNode, while the link only accepts BaseObjects. Should return false. But returns true, since findLinkDest returns false in read()
        EXPECT_FALSE(withOwner.read("@/")); 
    }


};

TEST_F( Link_test, setLinkedBase_test)
{
    this->setLinkedBase_test() ;
}

TEST_F( Link_test, read_test)
{
    this->read_test() ;
}

TEST_F(Link_test, DISABLED_read_test_tofix)
{
    this->read_test_tofix();
}

TEST_F( Link_test, read_multilink_test)
{
    this->read_multilink_test();
}

}// namespace sofa


