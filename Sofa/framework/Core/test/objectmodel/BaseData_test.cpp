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

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/core/objectmodel/BaseData.h>
using sofa::core::objectmodel::BaseData;

class MyData : public BaseData
{
public:
    MyData() : BaseData(BaseInitData()) {}
    bool read(const std::string&)override {return true;}
    void printValue(std::ostream&) const override {return;}
    std::string getValueString() const override  {return "";}
    std::string getDefaultValueString() const override { return ""; }
    std::string getValueTypeString() const override {return "";}
    const sofa::defaulttype::AbstractTypeInfo* getValueTypeInfo() const override {return nullptr;}
    const void* getValueVoidPtr() const {return nullptr;}
    void* beginEditVoidPtr(){return nullptr;}
    void* beginWriteOnlyVoidPtr(){return nullptr;}
    void endEditVoidPtr(){}
    bool doIsExactSameDataType(const BaseData* ) override{ return false; }
    bool doCopyValueFrom(const BaseData* ) override{ return false; }
    bool doSetValueFromLink(const BaseData* ) override{ return false; }
    const void* doGetValueVoidPtr() const override { return nullptr; }
    void* doBeginEditVoidPtr() override { return nullptr; }
    void doEndEditVoidPtr() override { }
};

class MyObject : public BaseObject
{
public:
    SOFA_CLASS(MyObject, BaseObject);
    MyData myData;
    MyObject() :
        myData()
    {
        myData.setName("myData");
        setName("node1");
        myData.setOwner(this);
    }
};

class BaseData_test: public BaseTest
{
public:
    MyObject m_object;
};

TEST_F(BaseData_test, setGetName)
{
    m_object.myData.setName("data1");
    ASSERT_EQ(m_object.myData.getName(), "data1");
}

TEST_F(BaseData_test, getLinkName)
{
    ASSERT_EQ(m_object.myData.getLinkPath(), "@node1.myData");
}
