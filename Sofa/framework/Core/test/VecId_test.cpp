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
#include <sofa/core/VecId.h>
#include <gtest/gtest.h>

TEST(VecId, name)
{
    auto position = sofa::core::VecCoordId::position();
    EXPECT_EQ(position.getName(), "position(V_COORD)");

    auto restPosition = sofa::core::VecCoordId::restPosition();
    EXPECT_EQ(restPosition.getName(), "restPosition(V_COORD)");

    auto freePosition = sofa::core::VecCoordId::freePosition();
    EXPECT_EQ(freePosition.getName(), "freePosition(V_COORD)");

    auto resetPosition = sofa::core::VecCoordId::resetPosition();
    EXPECT_EQ(resetPosition.getName(), "resetPosition(V_COORD)");



    auto velocity = sofa::core::VecDerivId::velocity();
    EXPECT_EQ(velocity.getName(), "velocity(V_DERIV)");

    auto resetVelocity = sofa::core::VecDerivId::resetVelocity();
    EXPECT_EQ(resetVelocity.getName(), "resetVelocity(V_DERIV)");

    auto freeVelocity = sofa::core::VecDerivId::freeVelocity();
    EXPECT_EQ(freeVelocity.getName(), "freeVelocity(V_DERIV)");

    auto normal = sofa::core::VecDerivId::normal();
    EXPECT_EQ(normal.getName(), "normal(V_DERIV)");

    auto force = sofa::core::VecDerivId::force();
    EXPECT_EQ(force.getName(), "force(V_DERIV)");

    auto externalForce = sofa::core::VecDerivId::externalForce();
    EXPECT_EQ(externalForce.getName(), "externalForce(V_DERIV)");

    auto dx = sofa::core::VecDerivId::dx();
    EXPECT_EQ(dx.getName(), "dx(V_DERIV)");

    auto dforce = sofa::core::VecDerivId::dforce();
    EXPECT_EQ(dforce.getName(), "dforce(V_DERIV)");

}
