/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * #include <SofaSimulationCommon/init.h>
using sofa::simulation::common::init ;

#include <SofaSimulationGraph/init.h>
using sofa::simulation::graph::init ;

#include <SofaComponentBase/initComponentBase.h>
using sofa::component::initComponentBase ;
*/

#include <sofa/core/objectmodel/BaseObjectDescription.h>
using sofa::core::objectmodel::BaseObjectDescription ;

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


struct BaseObjectDescription_test: public BaseTest
{
    void SetUp()
    {
    }

    void TearDown()
    {
    }

    void checkConstructorBehavior()
    {
        BaseObjectDescription objectDescription("theName", "theType");

        EXPECT_EQ( objectDescription.getName(), "theName");
        EXPECT_STREQ( objectDescription.getAttribute("name"), "theName");
        EXPECT_STREQ( objectDescription.getAttribute("type"), "theType");

        EXPECT_EQ( objectDescription.getBaseFile(), "") ;

        EXPECT_EQ( objectDescription.find(""), nullptr );
        EXPECT_EQ( objectDescription.find("aNonExistantName"), nullptr );

        EXPECT_EQ( objectDescription.getObject(), nullptr );

        /// This function is supposed to return an error message if there is no context.
        {
            EXPECT_MSG_EMIT(Error) ;
            EXPECT_EQ( objectDescription.findObject("aNonExistantName"), nullptr );
        }

    }

    void checkSetGetAttribute()
    {
        BaseObjectDescription objectDescription("theName", "theType");

        size_t numattr=objectDescription.getAttributeMap().size() ;

        objectDescription.setAttribute("anAttribute", "true") ;
        EXPECT_EQ( (objectDescription.getAttributeMap().size()), numattr+1) ;

        const char* res = objectDescription.getAttribute("anAttribute", "notFound") ;
        EXPECT_STREQ(res, "true");
    }

    void checkRemoveAnAttribute()
    {
        BaseObjectDescription objectDescription("theName", "theType");

        size_t numattr=objectDescription.getAttributeMap().size() ;

        objectDescription.setAttribute("anAttribute", "true") ;
        EXPECT_EQ( (objectDescription.getAttributeMap().size()), numattr+1) ;

        ASSERT_FALSE( objectDescription.removeAttribute("anAttributeThatDoesNotExist") );
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+1) ;

        ASSERT_TRUE( objectDescription.removeAttribute("anAttribute") ) ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr) ;

        const char* res = objectDescription.getAttribute("anAttribute", "notFound") ;
        EXPECT_STREQ(res, "notFound");
    }


    void checkGetAttributeAsFloat()
    {
        BaseObjectDescription objectDescription("theName", "theType");
        size_t numattr=objectDescription.getAttributeMap().size() ;

        objectDescription.setAttribute("anAttribute", "true") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+1) ;

        objectDescription.setAttribute("aFloatAttribute", "1.0") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+2) ;

        objectDescription.setAttribute("aFirstNonFloatAttribute", "1,0") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+3) ;

        objectDescription.setAttribute("aSecondNonFloatAttribute", "1.0 2.0") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+4) ;


        EXPECT_EQ( objectDescription.getAttributeAsFloat("anAttribute", -1234.0), -1234.0) ;
        EXPECT_EQ( objectDescription.getAttributeAsFloat("aFirstNonFloatAttribute", -1234.0), -1234.0) ;
        EXPECT_EQ( objectDescription.getAttributeAsFloat("aSecondNonFloatAttribute", -1234.0), -1234.0) ;
        EXPECT_EQ( objectDescription.getErrors().size(), (size_t)3) << "If this fails this means that one of the three previous "
                                                               "conversion succeded while it shouldn't";

        EXPECT_EQ( objectDescription.getAttributeAsFloat("aFloatAttribute", -1234.0), 1.0) ;
        EXPECT_EQ( objectDescription.getErrors().size(), (size_t)3) ;
    }

    void checkGetAttributeAsInt()
    {
        BaseObjectDescription objectDescription("theName", "theType");
        size_t numattr=objectDescription.getAttributeMap().size() ;

        objectDescription.setAttribute("anAttribute", "true") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+1) ;

        objectDescription.setAttribute("aFirstIntAttribute", "234") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+2) ;

        objectDescription.setAttribute("aSecondIntAttribute", "-234") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+2) ;

        objectDescription.setAttribute("aFirstNonIntAttribute", "1,0") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+3) ;

        objectDescription.setAttribute("aSecondNonIntAttribute", "1.0") ;
        EXPECT_EQ( objectDescription.getAttributeMap().size(), numattr+4) ;

        EXPECT_EQ( objectDescription.getAttributeAsInt("aFirstIntAttribute", 1234), 234) ;
        EXPECT_EQ( objectDescription.getAttributeAsInt("aSecondIntAttribute", 1234), -234) ;

        EXPECT_EQ( objectDescription.getAttributeAsInt("aFirstNonIntAttribute", -1234.0), -1234.0) ;
        EXPECT_EQ( objectDescription.getAttributeAsInt("aSecondNonIntAttribute", -1234.0), -1234.0) ;
        EXPECT_EQ( objectDescription.getErrors().size(), (size_t)2) << "If this fails this means that one of the three previous "
                                                               "conversion succeded while it shouldn't";
    }
};


TEST_F(BaseObjectDescription_test, checkConstructorBehavior)
{
    this->checkConstructorBehavior();
}

TEST_F(BaseObjectDescription_test, checkSetGetAttribute)
{
    this->checkSetGetAttribute();
}

TEST_F(BaseObjectDescription_test ,  checkGetAttributeAsFloat)
{
    this->checkGetAttributeAsFloat();
}

TEST_F(BaseObjectDescription_test ,  checkGetAttributeAsInt)
{
    this->checkGetAttributeAsFloat();
}

TEST_F(BaseObjectDescription_test ,  checkRemoveAnAttribute)
{
    this->checkRemoveAnAttribute();
}

