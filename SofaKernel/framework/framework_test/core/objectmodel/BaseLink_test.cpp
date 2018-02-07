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
 *    - damien.marchal@univ-lille1.fr
 *****************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/Link.h>
using sofa::core::objectmodel::SingleLink ;
using sofa::core::objectmodel::BaseLink ;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

class EmptyObject : public BaseObject
{
public:
    SOFA_CLASS(EmptyObject, BaseObject) ;
};

class SingleLink_test: public BaseTest
{
public:
    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_STRONGLINK > m_link ;
    BaseObject::SPtr m_dst ;
    BaseObject::SPtr m_src ;

    /// Create a link to an object.
    void SetUp()
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
    ASSERT_EQ(m_link.size(), (unsigned int)1) ;
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
    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_MULTILINK > mlink ;
    ASSERT_EQ(mlink.size(), (unsigned int)0) ;
    mlink.add(m_dst.get()) ;
    ASSERT_EQ(mlink.size(), (unsigned int)1) ;
    mlink.add(m_dst.get()) ;
    ASSERT_EQ(mlink.size(), (unsigned int)1) ;

    SingleLink<BaseObject, BaseObject, BaseLink::FLAG_NONE > slink ;
    ASSERT_EQ(slink.size(), (unsigned int)0) ;
    slink.add(m_dst.get()) ;
    ASSERT_EQ(slink.size(), (unsigned int)1) ;
    slink.add(m_dst.get()) ;
    ASSERT_EQ(slink.size(), (unsigned int)1) ;
}
