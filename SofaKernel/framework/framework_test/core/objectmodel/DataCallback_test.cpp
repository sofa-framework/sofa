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
#include <sofa/core/objectmodel/DataCallback.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest ;


namespace sofa {

using namespace core::objectmodel;


/**  Test suite for data callbacks
  */
struct DataCallback_test: public BaseTest
{
    class TestObject : public sofa::core::objectmodel::BaseObject
    {
    public:
        //data attached to an object
        Data<int> d_objdata1;
        Data<int> d_objdata2;
        DataCallback m_datacallback1;
        DataCallback m_datacallback2;
        DataCallback m_datacallbackAll;

        void printData1()
        {
            msg_info("DataCallback_test") << "TestObject : Value of objdata1 changed : "
                                          << this->d_objdata1.getValue();
            msg_warning("DataCallback_test") << "TestObject : Value of objdata2 did not changed : "
                                           << this->d_objdata2.getValue();
        }
        void printData2()
        {
            msg_advice("DataCallback_test") << "TestObject : Value of objdata2 changed : "
                                          << this->d_objdata2.getValue();
            msg_error("DataCallback_test") << "TestObject : Value of objdata1 did not changed : "
                                           << this->d_objdata1.getValue();
        }
        void printDataAll()
        {
            msg_fatal("DataCallback_test") << "TestObject : Value of objdata1 or objdata2 changed : "
                                          << this->d_objdata1.getValue() << " | "
                                          << this->d_objdata2.getValue();
        }

        TestObject()
            : sofa::core::objectmodel::BaseObject()
            , d_objdata1(initData(&d_objdata1, 0, "objdata1", "objdata1"))
            , d_objdata2(initData(&d_objdata2, 1, "objdata2", "objdata2"))
            , m_datacallback1(&d_objdata1)
            , m_datacallback2(&d_objdata2)
            , m_datacallbackAll( {&d_objdata1, &d_objdata2} )
        {
        }
    };

    void SetUp()
    {

    }

};

TEST_F(DataCallback_test, testDataCallback_1)
{
    TestObject obj;
    obj.m_datacallback1.addCallback(&TestObject::printData1);

    EXPECT_EQ( obj.d_objdata1.getValue(), 0 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 1 ) ;

    //callback is expected to print an info and a warning message
    EXPECT_MSG_EMIT(Info) ;
    EXPECT_MSG_EMIT(Warning) ;
    obj.d_objdata1.setValue(123);
    EXPECT_EQ( obj.d_objdata1.getValue(), 123 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 1 ) ;
}

TEST_F(DataCallback_test, testDataCallback_2)
{
    TestObject obj;
    obj.m_datacallback2.addCallback(&TestObject::printData2);

    EXPECT_EQ( obj.d_objdata1.getValue(), 0 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 1 ) ;

    //callback is expected to print an advice and an error message
    EXPECT_MSG_EMIT(Advice) ;
    EXPECT_MSG_EMIT(Error) ;
    obj.d_objdata2.setValue(456);
    EXPECT_EQ( obj.d_objdata1.getValue(), 0 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 456 ) ;
}

TEST_F(DataCallback_test, testDataCallback_All)
{
    TestObject obj;
    obj.m_datacallbackAll.addCallback(&TestObject::printDataAll);

    EXPECT_EQ( obj.d_objdata1.getValue(), 0 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 1 ) ;

    //callback is expected to print a fatal message
    EXPECT_MSG_EMIT(Fatal) ;
    obj.d_objdata1.setValue(234);
    EXPECT_EQ( obj.d_objdata1.getValue(), 234 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 1 ) ;
    EXPECT_MSG_EMIT(Fatal) ; // how to expect an other (fatal) message ?
    obj.d_objdata2.setValue(987);
    EXPECT_EQ( obj.d_objdata1.getValue(), 234 ) ;
    EXPECT_EQ( obj.d_objdata2.getValue(), 987 ) ;
}


}// namespace sofa
