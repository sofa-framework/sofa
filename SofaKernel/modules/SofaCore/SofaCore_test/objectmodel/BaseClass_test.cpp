/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
 * Contributors:                                                              *
 *    - apalliyali1@hamad.qa                                                  *
 *****************************************************************************/
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;



#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;

namespace sofa{
namespace another_namespace{

class EmptyObject : public BaseObject
{
public:
    SOFA_CLASS(EmptyObject, BaseObject) ;
};

}
}

namespace sofa{
namespace numbered_namespace_123{

class NumberedClass123 : public BaseObject
{
public:
    SOFA_CLASS(NumberedClass123, BaseObject) ;
};

}
}

class BaseClass_test: public BaseTest
{
public:
    sofa::another_namespace::EmptyObject* m_ptr1 = NULL ;
    sofa::numbered_namespace_123::NumberedClass123* m_ptr2 = NULL ;
    

    void SetUp() override
    {
    }
};

TEST_F(BaseClass_test, checkClassName  )
{
    ASSERT_STREQ(BaseObject::className(m_ptr1).c_str(),"EmptyObject") ;
    ASSERT_STREQ(BaseObject::className(m_ptr2).c_str(),"NumberedClass123") ;
}

TEST_F(BaseClass_test, checkNameSpace  )
{
    ASSERT_STREQ(BaseObject::namespaceName(m_ptr1).c_str(),"sofa::another_namespace") ;
    ASSERT_STREQ(BaseObject::namespaceName(m_ptr2).c_str(),"sofa::numbered_namespace_123") ;
}