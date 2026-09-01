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
#include <sofa/core/MultiVecId.h>
#include <sofa/core/BaseState.h>
#include <gtest/gtest.h>

#include <string>
#include <set>
#include <sstream>

namespace sofa
{

/// Concrete mock of BaseState to avoid undefined behavior or invalid pointer dereferences.
class MockBaseState : public core::BaseState
{
public:
    explicit MockBaseState(const std::string& name = "mockState")
    {
        setName(name);
    }

    Size getSize() const override { return 0; }
    void resize(Size) override {}
    core::objectmodel::BaseData* baseWrite(core::VecId) override { return nullptr; }
    const core::objectmodel::BaseData* baseRead(core::ConstVecId) const override { return nullptr; }
};

// ============================================================================
// 1. Generic & Template Specialization Standard Functionality Tests
// ============================================================================

TEST(MultiVecIdTest, DefaultConstructorAndNullForAllTypes)
{
    // V_COORD (Write & Read)
    core::MultiVecCoordId coordWrite;
    EXPECT_TRUE(coordWrite.isNull());
    EXPECT_EQ(coordWrite.getDefaultId(), core::VecCoordId::null());
    EXPECT_FALSE(coordWrite.hasIdMap());

    core::ConstMultiVecCoordId coordRead;
    EXPECT_TRUE(coordRead.isNull());
    EXPECT_EQ(coordRead.getDefaultId(), core::ConstVecCoordId::null());
    EXPECT_FALSE(coordRead.hasIdMap());

    // V_DERIV (Write & Read)
    core::MultiVecDerivId derivWrite;
    EXPECT_TRUE(derivWrite.isNull());
    EXPECT_EQ(derivWrite.getDefaultId(), core::VecDerivId::null());
    EXPECT_FALSE(derivWrite.hasIdMap());

    core::ConstMultiVecDerivId derivRead;
    EXPECT_TRUE(derivRead.isNull());
    EXPECT_EQ(derivRead.getDefaultId(), core::ConstVecDerivId::null());
    EXPECT_FALSE(derivRead.hasIdMap());

    // V_MATDERIV (Write & Read)
    core::MultiMatrixDerivId matDerivWrite;
    EXPECT_TRUE(matDerivWrite.isNull());
    EXPECT_EQ(matDerivWrite.getDefaultId(), core::MatrixDerivId::null());
    EXPECT_FALSE(matDerivWrite.hasIdMap());

    core::ConstMultiMatrixDerivId matDerivRead;
    EXPECT_TRUE(matDerivRead.isNull());
    EXPECT_EQ(matDerivRead.getDefaultId(), core::ConstMatrixDerivId::null());
    EXPECT_FALSE(matDerivRead.hasIdMap());

    // V_ALL (Write & Read)
    core::MultiVecId allWrite;
    EXPECT_TRUE(allWrite.isNull());
    EXPECT_EQ(allWrite.getDefaultId(), core::VecId::null());
    EXPECT_FALSE(allWrite.hasIdMap());

    core::ConstMultiVecId allRead;
    EXPECT_TRUE(allRead.isNull());
    EXPECT_EQ(allRead.getDefaultId(), core::ConstVecId::null());
    EXPECT_FALSE(allRead.hasIdMap());
}

TEST(MultiVecIdTest, ConstructorFromVecIdForAllTypes)
{
    // V_COORD
    const core::VecCoordId& posWrite = core::vec_id::write_access::position;
    const core::ConstVecCoordId& posRead = core::vec_id::read_access::position;

    {
        core::MultiVecCoordId multiCoordWrite(posWrite);
        EXPECT_FALSE(multiCoordWrite.isNull());
        EXPECT_EQ(multiCoordWrite.getDefaultId(), posWrite);
    }

    {
        // forbidden
        // core::MultiVecCoordId multiCoordWrite(posRead);
        // EXPECT_FALSE(multiCoordWrite.isNull());
        // EXPECT_EQ(multiCoordWrite.getDefaultId(), posRead);
    }

    {
        core::ConstMultiVecCoordId multiCoordRead(posWrite);
        EXPECT_FALSE(multiCoordRead.isNull());
        EXPECT_EQ(multiCoordRead.getDefaultId(), posWrite);
    }

    {
        core::ConstMultiVecCoordId multiCoordRead(posRead);
        EXPECT_FALSE(multiCoordRead.isNull());
        EXPECT_EQ(multiCoordRead.getDefaultId(), posRead);
    }

    // V_DERIV
    const core::VecDerivId& velWrite = core::vec_id::write_access::velocity;
    const core::ConstVecDerivId& velRead = core::vec_id::read_access::velocity;

    {
        core::MultiVecDerivId multiDerivWrite(velWrite);
        EXPECT_FALSE(multiDerivWrite.isNull());
        EXPECT_EQ(multiDerivWrite.getDefaultId(), velWrite);
    }

    {
        //forbidden
        // core::MultiVecDerivId multiDerivWrite(velRead);
        // EXPECT_FALSE(multiDerivWrite.isNull());
        // EXPECT_EQ(multiDerivWrite.getDefaultId(), velRead);
    }

    {
        core::ConstMultiVecDerivId multiDerivRead(velRead);
        EXPECT_FALSE(multiDerivRead.isNull());
        EXPECT_EQ(multiDerivRead.getDefaultId(), velRead);
    }

    {
        core::ConstMultiVecDerivId multiDerivRead(velWrite);
        EXPECT_FALSE(multiDerivRead.isNull());
        EXPECT_EQ(multiDerivRead.getDefaultId(), velWrite);
    }

    // V_MATDERIV
    core::MatrixDerivId matWrite = core::vec_id::write_access::constraintJacobian;
    core::ConstMatrixDerivId matRead = core::vec_id::read_access::constraintJacobian;

    {
        core::MultiMatrixDerivId multiMatWrite(matWrite);
        EXPECT_FALSE(multiMatWrite.isNull());
        EXPECT_EQ(multiMatWrite.getDefaultId(), matWrite);
    }

    {
        //forbidden
        // core::MultiMatrixDerivId multiMatWrite(matRead);
        // EXPECT_FALSE(multiMatWrite.isNull());
        // EXPECT_EQ(multiMatWrite.getDefaultId(), matRead);
    }

    {
        core::ConstMultiMatrixDerivId multiMatRead(matRead);
        EXPECT_FALSE(multiMatRead.isNull());
        EXPECT_EQ(multiMatRead.getDefaultId(), matRead);
    }

    {
        core::ConstMultiMatrixDerivId multiMatRead(matWrite);
        EXPECT_FALSE(multiMatRead.isNull());
        EXPECT_EQ(multiMatRead.getDefaultId(), matWrite);
    }

    // V_ALL
    {
        core::MultiVecId multiAllWrite(posWrite);
        EXPECT_EQ(multiAllWrite.getDefaultId(), posWrite);
    }

    {
        // forbidden
        // core::MultiVecId multiAllWrite(posRead);
        // EXPECT_EQ(multiAllWrite.getDefaultId(), posRead);
    }

    {
        core::ConstMultiVecId multiAllRead(posRead);
        EXPECT_EQ(multiAllRead.getDefaultId(), posRead);
    }

    {
        core::ConstMultiVecId multiAllRead(posWrite);
        EXPECT_EQ(multiAllRead.getDefaultId(), posWrite);
    }
}

TEST(MultiVecIdTest, SetIdAndGetIdPerStateForAllTypes)
{
    using namespace core::vec_id;

    MockBaseState state1("state1");
    MockBaseState state2("state2");
    MockBaseState unregisteredState("unregisteredState");

    // 1. MultiVecCoordId
    {
        core::MultiVecCoordId multi(write_access::position);
        multi.setId(&state1, write_access::freePosition);
        multi.setId(&state2, write_access::restPosition);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::freePosition);
        EXPECT_EQ(multi.getId(&state2), write_access::restPosition);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::position);

        multi.assign(write_access::resetPosition);
        EXPECT_FALSE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::resetPosition);
        EXPECT_EQ(multi.getId(&state2), write_access::resetPosition);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::resetPosition);
    }

    // 2. ConstMultiVecCoordId
    {
        core::ConstMultiVecCoordId multi(read_access::position);
        multi.setId(&state1, read_access::freePosition);
        multi.setId(&state2, read_access::restPosition);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), read_access::freePosition);
        EXPECT_EQ(multi.getId(&state2), read_access::restPosition);
        EXPECT_EQ(multi.getId(&unregisteredState), read_access::position);

        multi.assign(read_access::resetPosition);
        EXPECT_FALSE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::resetPosition);
        EXPECT_EQ(multi.getId(&state2), write_access::resetPosition);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::resetPosition);
    }

    // 3. MultiVecDerivId
    {
        core::MultiVecDerivId multi(write_access::velocity);
        multi.setId(&state1, write_access::force);
        multi.setId(&state2, write_access::freeVelocity);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::force);
        EXPECT_EQ(multi.getId(&state2), write_access::freeVelocity);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::velocity);

        multi.assign(write_access::dx);
        EXPECT_FALSE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::dx);
        EXPECT_EQ(multi.getId(&state2), write_access::dx);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::dx);
    }

    // 4. ConstMultiVecDerivId
    {
        core::ConstMultiVecDerivId multi(read_access::velocity);
        multi.setId(&state1, read_access::force);
        multi.setId(&state2, read_access::freeVelocity);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), read_access::force);
        EXPECT_EQ(multi.getId(&state2), read_access::freeVelocity);
        EXPECT_EQ(multi.getId(&unregisteredState), read_access::velocity);

        multi.assign(read_access::dx);
        EXPECT_FALSE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), read_access::dx);
        EXPECT_EQ(multi.getId(&state2), read_access::dx);
        EXPECT_EQ(multi.getId(&unregisteredState), read_access::dx);
    }

    // 5. MultiMatrixDerivId
    {
        core::MultiMatrixDerivId multi(write_access::constraintJacobian);
        multi.setId(&state1, write_access::mappingJacobian);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::mappingJacobian);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::constraintJacobian);
    }

    // 6. ConstMultiMatrixDerivId
    {
        core::ConstMultiMatrixDerivId multi(read_access::constraintJacobian);
        multi.setId(&state1, read_access::mappingJacobian);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), read_access::mappingJacobian);
        EXPECT_EQ(multi.getId(&unregisteredState), read_access::constraintJacobian);
    }

    // 7. MultiVecId (V_ALL, V_WRITE)
    {
        core::MultiVecId multi(write_access::position);
        multi.setId(&state1, write_access::velocity);
        multi.setId(&state2, write_access::constraintJacobian);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), write_access::velocity);
        EXPECT_EQ(multi.getId(&state2), write_access::constraintJacobian);
        EXPECT_EQ(multi.getId(&unregisteredState), write_access::position);
    }

    // 8. ConstMultiVecId (V_ALL, V_READ)
    {
        core::ConstMultiVecId multi(read_access::position);
        multi.setId(&state1, read_access::velocity);
        multi.setId(&state2, read_access::constraintJacobian);

        EXPECT_TRUE(multi.hasIdMap());
        EXPECT_EQ(multi.getId(&state1), read_access::velocity);
        EXPECT_EQ(multi.getId(&state2), read_access::constraintJacobian);
        EXPECT_EQ(multi.getId(&unregisteredState), read_access::position);
    }
}

TEST(MultiVecIdTest, IsNullWithIdMapForAllTypes)
{
    MockBaseState state("state");

    // V_COORD
    core::MultiVecCoordId coord(core::VecCoordId::null());
    EXPECT_TRUE(coord.isNull());
    coord.setId(&state, core::vec_id::write_access::position);
    EXPECT_FALSE(coord.isNull());
    coord.setId(&state, core::VecCoordId::null());
    EXPECT_TRUE(coord.isNull());

    // V_DERIV
    core::MultiVecDerivId deriv(core::VecDerivId::null());
    EXPECT_TRUE(deriv.isNull());
    deriv.setId(&state, core::vec_id::write_access::velocity);
    EXPECT_FALSE(deriv.isNull());
    deriv.setId(&state, core::VecDerivId::null());
    EXPECT_TRUE(deriv.isNull());

    // V_MATDERIV
    core::MultiMatrixDerivId mat(core::MatrixDerivId::null());
    EXPECT_TRUE(mat.isNull());
    mat.setId(&state, core::vec_id::write_access::constraintJacobian);
    EXPECT_FALSE(mat.isNull());
    mat.setId(&state, core::MatrixDerivId::null());
    EXPECT_TRUE(mat.isNull());

    // V_ALL
    core::MultiVecId all(core::VecId::null());
    EXPECT_TRUE(all.isNull());
    all.setId(&state, core::vec_id::write_access::position);
    EXPECT_FALSE(all.isNull());
    all.setId(&state, core::VecId::null());
    EXPECT_TRUE(all.isNull());
}

TEST(MultiVecIdTest, GetNameAndStreamingWithValidMockState)
{
    MockBaseState state1("state1");
    MockBaseState state2("state2");

    core::MultiVecCoordId coord(core::vec_id::write_access::position);
    EXPECT_EQ(coord.getName(), core::vec_id::write_access::position.getName());

    coord.setId(&state1, core::vec_id::write_access::freePosition);
    coord.setId(&state2, core::vec_id::write_access::freePosition);

    std::string name = coord.getName();
    EXPECT_EQ(name.front(), '{');
    EXPECT_EQ(name.back(), '}');
    EXPECT_NE(name.find("state1"), std::string::npos);
    EXPECT_NE(name.find("state2"), std::string::npos);

    std::ostringstream ss;
    ss << coord;
    EXPECT_EQ(ss.str(), name);
}

TEST(MultiVecIdTest, AccessConversionCompatibility)
{
    core::MultiVecCoordId writeCoord(core::vec_id::write_access::position);
    core::ConstMultiVecCoordId readCoord(writeCoord);
    EXPECT_EQ(readCoord.getDefaultId(), writeCoord.getDefaultId());

    core::MultiVecDerivId writeDeriv(core::vec_id::write_access::velocity);
    core::ConstMultiVecDerivId readDeriv(writeDeriv);
    EXPECT_EQ(readDeriv.getDefaultId(), writeDeriv.getDefaultId());

    core::MultiMatrixDerivId writeMat(core::vec_id::write_access::constraintJacobian);
    core::ConstMultiMatrixDerivId readMat(writeMat);
    EXPECT_EQ(readMat.getDefaultId(), writeMat.getDefaultId());

    core::MultiVecId writeAll(core::vec_id::write_access::position);
    core::ConstMultiVecId readAll(writeAll);
    EXPECT_EQ(readAll.getDefaultId(), writeAll.getDefaultId());
}

TEST(MultiVecIdTest, GetNameWithAllocatedEmptyMap)
{
    core::MultiVecCoordId coord(core::vec_id::write_access::position);

    std::set<const core::BaseState*> emptySet;
    coord.setId(emptySet, core::vec_id::write_access::freePosition);
    EXPECT_FALSE(coord.hasIdMap());

    EXPECT_EQ(coord.getName(), core::vec_id::write_access::position.getName());
}

} // namespace sofa
