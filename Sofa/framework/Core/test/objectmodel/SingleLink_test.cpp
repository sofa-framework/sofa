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
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/fwd.h>
using sofa::core::objectmodel::BaseObject ;
#include <sofa/simulation/Node.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/objectmodel/Link.h>
using sofa::core::objectmodel::SingleLink ;
using sofa::core::objectmodel::BaseLink ;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

#include "BaseLink_test.h"

class EmptyObject : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(EmptyObject, sofa::core::objectmodel::BaseObject) ;
};

class SingleLink_test: public BaseTest
{
public:
    SingleLink<sofa::core::objectmodel::BaseObject, sofa::core::objectmodel::BaseObject, BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK|BaseLink::FLAG_STOREPATH > m_link ;
    sofa::core::objectmodel::BaseObject::SPtr m_dst ;
    sofa::core::objectmodel::BaseObject::SPtr m_src ;
    sofa::simulation::Node::SPtr m_root { nullptr };

    /// Create a link to an object.
    void doSetUp() override
    {
        m_dst = sofa::core::objectmodel::New<sofa::core::objectmodel::BaseObject>() ;
        m_src = sofa::core::objectmodel::New<sofa::core::objectmodel::BaseObject>() ;

        m_dst->setName("destination") ;
        m_src->setName("source") ;
        m_src->addLink(&m_link);
        m_link.add(m_dst.get());
    }

    void setupContext()
    {
        m_root = sofa::simulation::getSimulation()->createNewGraph("root");

        m_root->addObject(this->m_src);
        m_root->addObject(this->m_dst);

        m_link.set(nullptr);
        ASSERT_EQ(m_link.get(), nullptr);

        m_src->addLink(&m_link);
        m_link.setOwner(m_src.get());
    }
};

TEST_F(SingleLink_test, checkAccess  )
{
    ASSERT_EQ(m_link.get(), m_dst.get()) ;
    ASSERT_FALSE(m_link.empty()) ;
    ASSERT_EQ(m_link.size(), size_t(1)) ;
}

TEST_F(SingleLink_test, checkIsSetPersistent  )
{
    m_link.setPersistent(true) ;
    ASSERT_TRUE( m_link.isPersistent() ) ;
    m_link.setPersistent(false) ;
    ASSERT_FALSE( m_link.isPersistent() ) ;
}

TEST_F(SingleLink_test, checkCounterLogic )
{
    ASSERT_EQ(m_link.getCounter(),1) ;
    m_link.reset() ;
    ASSERT_EQ(m_link.getCounter(),2) ;
    m_link.add(m_dst.get()) ;
    ASSERT_EQ(m_link.getCounter(),3) ;
    m_link.add(m_dst.get()) ;
    ASSERT_EQ(m_link.getCounter(),4) ;
}

TEST_F(SingleLink_test, checkMultiLink )
{
    SingleLink<sofa::core::objectmodel::BaseObject, sofa::core::objectmodel::BaseObject, BaseLink::FLAG_MULTILINK > smlink ;
    ASSERT_EQ(smlink.size(), size_t(0)) ;
    smlink.add(m_dst.get()) ;
    ASSERT_EQ(smlink.size(), size_t(1)) ;
    smlink.add(m_dst.get()) ;
    ASSERT_EQ(smlink.size(), size_t(1)) ;

    SingleLink<sofa::core::objectmodel::BaseObject, sofa::core::objectmodel::BaseObject, BaseLink::FLAG_NONE > slink ;
    ASSERT_EQ(slink.size(), size_t(0)) ;
    slink.add(m_dst.get()) ;
    ASSERT_EQ(slink.size(), size_t(1)) ;
    slink.add(m_dst.get()) ;
    ASSERT_EQ(slink.size(), size_t(1)) ;
}

TEST_F(SingleLink_test, getOwnerBase)
{
    const auto aBaseObject = sofa::core::objectmodel::New<sofa::core::objectmodel::BaseObject>();
    using sofa::core::objectmodel::BaseNode;
    const BaseLink::InitLink<sofa::core::objectmodel::BaseObject> initObjectLink(aBaseObject.get(), "objectlink", "");
    const SingleLink<sofa::core::objectmodel::BaseObject, sofa::core::objectmodel::BaseObject, BaseLink::FLAG_NONE > objectLink(initObjectLink) ;
    ASSERT_EQ(objectLink.getOwnerBase(), aBaseObject.get());
    // m_link is initialized without an owner.
    // getOwnerBase() should still work and return a nullptr
    ASSERT_EQ(m_link.getOwnerBase(), nullptr);

    m_link.setOwner(aBaseObject.get());
    ASSERT_EQ(m_link.getOwnerBase(), aBaseObject.get());
}

TEST_F(SingleLink_test, checkClearSetValue  )
{
    m_link.clear();
    ASSERT_EQ( m_link.size(), 0 ) << "The size of a link container should be zero after clear().";
    m_link.set(nullptr);
    ASSERT_EQ( m_link.size(), 1 ) << "The size of a link container should be one after set(nullptr).";
    ASSERT_EQ( m_link.getLinkedPath(), "" ) << "The path should be empty because of the previously used set(nullptr).";
}

TEST_F(SingleLink_test, checkClearSetPath  )
{
    m_link.clear();
    ASSERT_EQ( m_link.size(), 0 )  << "The size of a link container should be zero after clear().";
    m_link.setPath("@/ThisIsAPath");
    ASSERT_EQ( m_link.size(), 1 ) << "The size of a link container should be one after setPath().";
    ASSERT_EQ( m_link.getLinkedPath(), "@/ThisIsAPath" ) << "The path should not be empty as it was set previously.";
}

TEST_F(SingleLink_test, checkEmptyness  )
{
    m_link.set(nullptr);
    ASSERT_EQ( m_link.size(), 1 );
    m_link.clear();
    ASSERT_EQ( m_link.size(), 0 );
}

TEST_F(SingleLink_test, readWithoutContext)
{
    m_link.set(nullptr);
    ASSERT_EQ(m_link.get(), nullptr);

    const bool success  = m_link.read("@/destination");
    ASSERT_FALSE(success);
    ASSERT_EQ(m_link.get(), nullptr);
}



TEST_F(SingleLink_test, readWithContextEmptyString)
{
    setupContext();

    const bool success  = m_link.read("");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), nullptr);
}

TEST_F(SingleLink_test, readWithContextJustAArobase)
{
    setupContext();

    // the path points to an invalid object, but it's ok
    const bool success  = m_link.read("@");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), nullptr);
}

TEST_F(SingleLink_test, readWithContextValidLink)
{
    setupContext();

    const bool success  = m_link.read("@/destination");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), m_dst.get());
}

TEST_F(SingleLink_test, readWithContextInvalidLink)
{
    setupContext();

    // the path points to an invalid object, but it's ok
    const bool success  = m_link.read("@/foo");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), nullptr);
}

TEST_F(SingleLink_test, readWithContextValidLinkDoubleSlash)
{
    setupContext();

    const bool success  = m_link.read("@//destination");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), m_dst.get());
}

TEST_F(SingleLink_test, readWithContextValidLinkNoSlash)
{
    setupContext();

    const bool success  = m_link.read("@destination");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), m_dst.get());
}

TEST_F(SingleLink_test, readWithContextValidLinkLeadingSpaces)
{
    setupContext();

    const bool success  = m_link.read("     @/destination");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), m_dst.get());
    ASSERT_EQ(m_link.getSize(), 1);
}

TEST_F(SingleLink_test, readWithContextValidLinkTrailingSpaces)
{
    setupContext();

    const bool success  = m_link.read("@/destination    ");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), m_dst.get());
    ASSERT_EQ(m_link.getSize(), 1);
}

TEST_F(SingleLink_test, readWithContextValidLinkTrailingSpacesButTheyArePartOfThePath)
{
    setupContext();

    m_dst->setName("destination    ");

    const bool success  = m_link.read("@/destination    ");
    ASSERT_TRUE(success);
    ASSERT_EQ(m_link.get(), m_dst.get());
    ASSERT_EQ(m_link.getSize(), 1);
}

TEST_F(SingleLink_test, readWithContextWithDot)
{
    setupContext();
    m_dst->setName("Component.With.Dots");

    const bool success  = m_link.read("@Component.With.Dots");
    ASSERT_TRUE(success);
}

TEST_F(SingleLink_test, readWithContextWithSpace)
{
    setupContext();
    m_dst->setName("Backward Euler ODE Solver");

    const bool success  = m_link.read("@Backward Euler ODE Solver");
    ASSERT_TRUE(success);
}

TEST_F(SingleLink_test, readComplex)
{
    m_root = sofa::simulation::getSimulation()->createNewGraph("root");
    m_root->addObject(this->m_src);

    auto child1 = m_root->createChild("child 1");
    auto child2 = child1->createChild("child 2");
    child2->addObject(this->m_dst);
    m_dst->setName("My Object");

    m_link.set(nullptr);
    ASSERT_EQ(m_link.get(), nullptr);

    m_src->addLink(&m_link);
    m_link.setOwner(m_src.get());

    {
        const bool success  = m_link.read("@/child 1/child 2/My Object");
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), m_dst.get());
        ASSERT_EQ(m_link.getSize(), 1);
    }

    {
        const bool success  = m_link.read("@/child 2/My Object");
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), nullptr);
        ASSERT_EQ(m_link.getSize(), 1);
    }

    {
        const bool success  = m_link.read("@/child 1/child 2/../child 2/My Object");
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), m_dst.get());
        ASSERT_EQ(m_link.getSize(), 1);
    }
}

TEST_F(SingleLink_test, readComplex2)
{
    m_root = sofa::simulation::getSimulation()->createNewGraph("root");

    auto child1 = m_root->createChild("child 1");
    child1->addObject(this->m_src);

    auto child2 = child1->createChild("child 2");
    child2->addObject(this->m_dst);
    m_dst->setName("My Object");

    m_link.set(nullptr);
    ASSERT_EQ(m_link.get(), nullptr);

    m_src->addLink(&m_link);
    m_link.setOwner(m_src.get());

    {
        const bool success  = m_link.read("@/child 1/child 2/My Object");
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), m_dst.get());
        ASSERT_EQ(m_link.getSize(), 1);
    }

    {
        const bool success  = m_link.read("@child 2/My Object");
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), m_dst.get());
        ASSERT_EQ(m_link.getSize(), 1);
    }

    {
        const bool success  = m_link.read("@child 2/My Object  "); //trailing spaces
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), m_dst.get());
        ASSERT_EQ(m_link.getSize(), 1);
    }

    {
        const bool success  = m_link.read("@/../child 1/child 2/My Object"); //ill-formed
        ASSERT_TRUE(success);
        ASSERT_EQ(m_link.get(), nullptr);
        ASSERT_EQ(m_link.getSize(), 1);
    }
}
