/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;
using testing::Types;

#include <sofa/helper/vector.h>
using sofa::helper::vector ;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data ;

#include <sofa/helper/logging/CountingMessageHandler.h>
using sofa::helper::logging::MainCountingMessageHandler ;
using sofa::helper::logging::CountingMessageHandler ;
using sofa::helper::logging::MessageDispatcher ;
using sofa::helper::logging::Message ;

template<class T>
class vector_test : public Sofa_test<>
{
public:
    void SetUp() {}
    void TearDown() {}
    void checkBasicReadFromString(const std::string& values) ;
    void checkIntervalReadFromString(const std::string& values,
                                     const std::string& results) ;
    void checkVectorDataField() ;
};

template<class T>
void vector_test<T>::checkBasicReadFromString(const std::string& values)
{
    vector<T> v;
    std::stringstream in(values);
    std::stringstream out ;

    CountingMessageHandler& counter = MainCountingMessageHandler::getInstance() ;
    int numErrors = counter.getMessageCountFor(Message::Warning) ;

    MessageDispatcher::addHandler( &counter ) ;
    v.read(in) ;
    v.write(out) ;
    /// If the parsed version is different that the written version & there is no warning...this
    /// means a problem will be un-noticed.
    if( in.str() != out.str() && counter.getMessageCountFor(Message::Warning) == numErrors ){
        FAIL() << "Input string [" << in.str() << "] return this vector [" << out.str() << "]"  ;
    }
    MessageDispatcher::rmHandler( &counter ) ;
}

template<class T>
void vector_test<T>::checkIntervalReadFromString(const std::string& values,
                                                 const std::string& results)
{
    std::cout << "Input: " << values << std::endl ;
    vector<T> v;
    std::stringstream in(values);
    std::stringstream out ;

    CountingMessageHandler& counter = MainCountingMessageHandler::getInstance() ;

    MessageDispatcher::addHandler( &counter ) ;
    v.read(in) ;
    v.write(out) ;
    /// If the parsed version is different that the written version & there is no warning...this
    /// means a problem will be un-noticed.
    if( out.str() != results ){
        FAIL() << "Input string [" << in.str() << "] return this vector [" << out.str() << "] while ["<< results << "]"  ;
    }
    MessageDispatcher::rmHandler( &counter ) ;
}

template<class T>
void vector_test<T>::checkVectorDataField()
{
    /*
    Data<RGBAColor> color ;

    EXPECT_FALSE(color.read("invalidcolor"));

    EXPECT_TRUE(color.read("white"));
    EXPECT_EQ(color.getValue(), RGBAColor(1.0,1.0,1.0,1.0));

    EXPECT_TRUE(color.read("blue"));
    EXPECT_EQ(color.getValue(), RGBAColor(0.0,0.0,1.0,1.0));

    std::stringstream tmp;
    tmp << color ;
    EXPECT_TRUE(color.read(tmp.str()));
    EXPECT_EQ(color.getValue(), RGBAColor(0.0,0.0,1.0,1.0));
    */
}

typedef Types<
   int,
   char,
   float,
   double,
   unsigned char,
   unsigned int
> DataTypes;
TYPED_TEST_CASE(vector_test, DataTypes);

TYPED_TEST(vector_test, checkBasicReadFromString_OpenIssue)
{
    this->checkBasicReadFromString("0 1 2 3 4 5 6") ;
    this->checkBasicReadFromString("zero un deux trois quatre cinq six") ;
    this->checkBasicReadFromString("0  un deux trois 4 5 6") ;
    this->checkBasicReadFromString("0 1 -2 3 4 -5 6") ;
    this->checkBasicReadFromString("") ;
    this->checkBasicReadFromString("0") ;
    this->checkBasicReadFromString("-10") ;
    this->checkBasicReadFromString("5 10 - 66") ;
    this->checkBasicReadFromString("3.14 3.15 3.16") ;
}


TYPED_TEST(vector_test, checkIntervalReadFromString_OpenIssue)
{
    this->checkIntervalReadFromString("10-20", "10 11 12 13 14 15 16 17 18 19 20") ;
    this->checkIntervalReadFromString("20-10", "20 19 18 17 16 15 14 13 12 11 10") ;

    /*
    this->checkIntervalReadFromString("10--20", "") ;
    this->checkIntervalReadFromString("1--0", "") ;

    this->checkIntervalReadFromString("--0", "") ;
    this->checkIntervalReadFromString("-0", "") ;
    this->checkIntervalReadFromString("0--", "") ;
    this->checkIntervalReadFromString("0-", "") ;
    this->checkIntervalReadFromString("10-", "") ;
    this->checkIntervalReadFromString("-", "") ;
    */
}

TYPED_TEST(vector_test, checkVectorDataField)
{
    this->checkVectorDataField() ;
}
