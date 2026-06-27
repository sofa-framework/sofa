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
#include <gtest/gtest.h>
#include <sofa/core/ComponentFactory.h>
#include <sofa/testing/TestMessageHandler.h>

namespace sofa
{

struct DummyComponent : public core::objectmodel::BaseComponent
{};

template<class T>
struct DummyComponentWith1Template : public core::objectmodel::BaseComponent
{};

template<class T1, class T2>
struct DummyComponentWith2Template : public core::objectmodel::BaseComponent
{};

TEST(ComponentFactory, EmptyRegistrationData)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);

    core::ComponentFactory factory;
    factory.registerComponent<DummyComponent>(core::ComponentRegistrationDataBuilder());
}

}
