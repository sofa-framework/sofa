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
#include <sofa/Modules.h>

class DerivedBaseVecId : public sofa::core::BaseVecId
{
public:
    constexpr DerivedBaseVecId(sofa::core::VecType t, unsigned int i) : sofa::core::BaseVecId(t, i) {}
};

TEST(BaseVecId, constructor)
{
    static constexpr DerivedBaseVecId v(sofa::core::VecType::V_COORD, 4);
    EXPECT_EQ(v.getIndex(), 4);
    EXPECT_EQ(v.getType(), sofa::core::VecType::V_COORD);
}

TEST(VecId, name)
{
    static constexpr auto position = sofa::core::VecCoordId::position();
    EXPECT_EQ(position.getName(), "position(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::position.getName(), "position(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::position.getName(), "position(V_COORD)");

    static constexpr auto restPosition = sofa::core::VecCoordId::restPosition();
    EXPECT_EQ(restPosition.getName(), "restPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::restPosition.getName(), "restPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::restPosition.getName(), "restPosition(V_COORD)");

    static constexpr auto freePosition = sofa::core::VecCoordId::freePosition();
    EXPECT_EQ(freePosition.getName(), "freePosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::freePosition.getName(), "freePosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::freePosition.getName(), "freePosition(V_COORD)");

    static constexpr auto resetPosition = sofa::core::VecCoordId::resetPosition();
    EXPECT_EQ(resetPosition.getName(), "resetPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::resetPosition.getName(), "resetPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::resetPosition.getName(), "resetPosition(V_COORD)");


    static constexpr auto velocity = sofa::core::VecDerivId::velocity();
    EXPECT_EQ(velocity.getName(), "velocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::velocity.getName(), "velocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::velocity.getName(), "velocity(V_DERIV)");

    static constexpr auto resetVelocity = sofa::core::VecDerivId::resetVelocity();
    EXPECT_EQ(resetVelocity.getName(), "resetVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::resetVelocity.getName(), "resetVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::resetVelocity.getName(), "resetVelocity(V_DERIV)");

    static constexpr auto freeVelocity = sofa::core::VecDerivId::freeVelocity();
    EXPECT_EQ(freeVelocity.getName(), "freeVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::freeVelocity.getName(), "freeVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::freeVelocity.getName(), "freeVelocity(V_DERIV)");

    static constexpr auto normal = sofa::core::VecDerivId::normal();
    EXPECT_EQ(normal.getName(), "normal(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::normal.getName(), "normal(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::normal.getName(), "normal(V_DERIV)");

    static constexpr auto force = sofa::core::VecDerivId::force();
    EXPECT_EQ(force.getName(), "force(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::force.getName(), "force(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::force.getName(), "force(V_DERIV)");

    static constexpr auto externalForce = sofa::core::VecDerivId::externalForce();
    EXPECT_EQ(externalForce.getName(), "externalForce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::externalForce.getName(), "externalForce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::externalForce.getName(), "externalForce(V_DERIV)");

    static constexpr auto dx = sofa::core::VecDerivId::dx();
    EXPECT_EQ(dx.getName(), "dx(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::dx.getName(), "dx(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::dx.getName(), "dx(V_DERIV)");

    static constexpr auto dforce = sofa::core::VecDerivId::dforce();
    EXPECT_EQ(dforce.getName(), "dforce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::dforce.getName(), "dforce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::dforce.getName(), "dforce(V_DERIV)");

    const std::string s = Sofa.Component.Collision;
    EXPECT_EQ(s, std::string("Sofa.Component.Collision"));

}
