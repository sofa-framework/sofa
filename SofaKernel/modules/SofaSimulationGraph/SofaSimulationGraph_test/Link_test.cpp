#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
using sofa::helper::testing::BaseSimulationTest ;

#include <SofaSimulationGraph/SimpleApi.h>
using namespace sofa::simpleapi ;

#include <sofa/simulation/testing/Node_test.h>

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
        ASSERT_EQ(objectLink.getLinkedBase(), aBaseObject.get());

        // EXPECT_MSG_EMIT(Error);
        nodeLink.setLinkedBase(aBasePtr); //< should emit error because BaseNode template type is incompatible with aBasePtr which is a BaseObject. But read() isn't implemented that way...

        ASSERT_NE(nodeLink.getLinkedBase(), aBasePtr);
    }

    void read_multilink_test()
    {
        SceneInstance si("root") ;
        BaseObject::SPtr A = sofa::core::objectmodel::New<BaseObject>();
        BaseObject::SPtr B = sofa::core::objectmodel::New<BaseObject>();
        BaseObject::SPtr C = sofa::core::objectmodel::New<BaseObject>();
        si.root->addObject(A);
        si.root->addObject(B);

        BaseLink::InitLink<BaseObject> il1(B.get(), "l1", "");
        MultiLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > withOwner(il1) ;

        // 1. test with valid link & owner
        ASSERT_TRUE(withOwner.read("@/B"));

        // 2. setting C's context
        si.root->addObject(B);

        ASSERT_TRUE(withOwner.read("@/C"));
        ASSERT_TRUE(withOwner.read("@/B @/C"));
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
        ASSERT_FALSE(withoutOwner.read("@/B")); // should return false as link has no owner

        // 2. test with valid link but no owner
        ASSERT_FALSE(withoutOwner.read("@"+A->getPathName())); // should return false as we have no owner to call findLinkDest with

        // 3. test with valid link & valid owner but no context
        ASSERT_TRUE(withOwner.read("@"+A->getPathName())); // should return true as the owner could be added later in the graph

        // setting B's context
        si.root->addObject(B);

        // 4. test with invalid link but valid owner
        ASSERT_FALSE(withOwner.read("/A")); // should return false as the link is invalid (should start with '@')
        ASSERT_TRUE(withOwner.read("@/plop")); // same as 3: plop could be added later in the graph, after init()
        ASSERT_FALSE(withOwner.read("@/\\!-#"))  << "read doesn't check path consistency, except for the presence of the '@'sign in the first character. This will currently return true";
        ASSERT_TRUE(withOwner.read("@/")); // Here link is OK, but points to a BaseNode, while the link only accepts BaseObjects. Should return false. But returns true, since findLinkDest returns false in read()

        // test with valid link & valid owner
        ASSERT_TRUE(withOwner.read("@/A")); // standard call: everything is initialized, link is OK, owner exists and has a context
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

TEST_F( Link_test, read_multilink_test)
{
    this->read_multilink_test();
}

}// namespace sofa


