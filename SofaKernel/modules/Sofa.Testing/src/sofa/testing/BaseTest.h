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
#pragma once

#include <sofa/testing/config.h>

#include <gtest/gtest.h>
#include <sofa/testing/TestMessageHandler.h>

namespace sofa::testing
{
/// acceptable ratio between finite difference delta and error threshold
const SReal g_minDeltaErrorRatio = .1;

/** @brief Base class for Sofa test fixtures.
  */
class SOFA_TESTING_API BaseTest : public ::testing::Test
{
public:
    /// To prevent that you simply need to add the line
    /// EXPECT_MSG_EMIT(Error); Where you want to allow a message.
    sofa::testing::MessageAsTestFailure m_fatal ;
    sofa::testing::MessageAsTestFailure m_error ;

    /// Initialize Sofa and the random number generator
    BaseTest() ;
    ~BaseTest() override;

    virtual void onSetUp() {}
    virtual void onTearDown() {}

    /// Seed value
    static int seed;

private:
    void SetUp() override ;
    void TearDown() override ;
};

} // namespace sofa::testing
