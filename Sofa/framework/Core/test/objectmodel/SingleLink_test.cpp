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
using sofa::core::objectmodel::BaseObject ;
#include <sofa/core/objectmodel/BaseNode.h>
using sofa::core::objectmodel::BaseNode ;

#include <sofa/core/objectmodel/Link.h>
using sofa::core::objectmodel::SingleLink ;
using sofa::core::objectmodel::BaseLink ;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

#include "BaseLink_test.h"

class EmptyObject : public BaseObject
{
public:
    SOFA_CLASS(EmptyObject, BaseObject) ;
};

class SingleLink_test: public BaseTest
{
public:
    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK|BaseLink::FLAG_STOREPATH > m_link ;
    BaseObject::SPtr m_dst ;
    BaseObject::SPtr m_src ;

    /// Create a link to an object.
    void SetUp() override
    {
        m_dst = sofa::core::objectmodel::New<BaseObject>() ;
        m_src = sofa::core::objectmodel::New<BaseObject>() ;

        m_dst->setName("destination") ;
        m_src->setName("source") ;
        m_src->addLink(&m_link);
        m_link.add(m_dst.get());
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
    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_MULTILINK > smlink ;
    ASSERT_EQ(smlink.size(), size_t(0)) ;
    smlink.add(m_dst.get()) ;
    ASSERT_EQ(smlink.size(), size_t(1)) ;
    smlink.add(m_dst.get()) ;
    ASSERT_EQ(smlink.size(), size_t(1)) ;

    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > slink ;
    ASSERT_EQ(slink.size(), size_t(0)) ;
    slink.add(m_dst.get()) ;
    ASSERT_EQ(slink.size(), size_t(1)) ;
    slink.add(m_dst.get()) ;
    ASSERT_EQ(slink.size(), size_t(1)) ;
}

TEST_F(SingleLink_test, getOwnerBase)
{
    const auto aBaseObject = sofa::core::objectmodel::New<BaseObject>();
    using sofa::core::objectmodel::BaseNode;
    const BaseLink::InitLink<BaseObject> initObjectLink(aBaseObject.get(), "objectlink", "");
    const SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > objectLink(initObjectLink) ;
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
