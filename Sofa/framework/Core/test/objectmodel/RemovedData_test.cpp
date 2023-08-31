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

#include <sofa/core/objectmodel/RemovedData.h>
using sofa::core::objectmodel::lifecycle::DeprecatedData;
using sofa::core::objectmodel::lifecycle::RemovedData;

namespace
{

class MyObject : public BaseObject
{
public:
    SOFA_CLASS(MyObject, BaseObject);

    DeprecatedData deprecatedData {this, "v23.06", "v23.12", "deprecatedData", "You should now use XXXX"};
    RemovedData removedData {this, "v23.06", "v23.12", "removedData", "You should now use XXXX"};
};

class RemoveData_test: public BaseTest
{
public:
    MyObject m_object;
};

TEST_F(RemoveData_test, testRemoved)
{
    EXPECT_MSG_EMIT(Error);
    EXPECT_MSG_NOEMIT(Deprecated);

    sofa::core::objectmodel::BaseObjectDescription desc;
    desc.setAttribute("removedData", "one");

    m_object.parse(&desc);
}

TEST_F(RemoveData_test, testDeprecated)
{
    EXPECT_MSG_EMIT(Deprecated);
    EXPECT_MSG_NOEMIT(Error);

    sofa::core::objectmodel::BaseObjectDescription desc;
    desc.setAttribute("deprecatedData", "one");

    m_object.parse(&desc);
}

}

