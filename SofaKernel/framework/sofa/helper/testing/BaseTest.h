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
/******************************************************************************
 * Contributors:
 *    - françois.faure
 *    - damien.marchal@univ-lille1.fr
 ******************************************************************************/
#ifndef SOFA_BASETEST_H
#define SOFA_BASETEST_H

#include <gtest/gtest.h>
#include <sofa/helper/testing/TestMessageHandler.h>

namespace sofa {
namespace helper {
namespace testing {
/// acceptable ratio between finite difference delta and error threshold
const SReal g_minDeltaErrorRatio = .1;

/** @brief Base class for Sofa test fixtures.
  */
class SOFA_HELPER_API BaseTest : public ::testing::Test
{
public:
    /// To prevent that you simply need to add the line
    /// EXPECT_MSG_EMIT(Error); Where you want to allow a message.
    sofa::helper::logging::MessageAsTestFailure m_fatal ;
    sofa::helper::logging::MessageAsTestFailure m_error ;

    /// Initialize Sofa and the random number generator
    BaseTest() ;
    virtual ~BaseTest();

    virtual void onSetUp() {}
    virtual void onTearDown() {}

    /// Seed value
    static int seed;

private:
    virtual void SetUp() override ;
    virtual void TearDown() override ;
};

} /// namespace testing
} /// namespace helper
} /// namespace sofa


#endif // SOFA_BASETEST_H
