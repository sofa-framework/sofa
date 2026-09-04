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
#include <sofa/core/State.inl>
#include <gtest/gtest.h>

#include <string>
#include <set>
#include <sstream>
#include <type_traits>

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
// 0. Compile-time conversion and assignment contract
//
// TMultiVecId's API surface is mostly a type system: which conversions are
// implicit, which must stay explicit, and which must not exist at all. None of
// that is observable from a runtime EXPECT, so it is pinned here. A refactor
// that silently drops one of these overloads breaks downstream code without
// touching a single test body.
// ============================================================================

// A TVecId converts implicitly into the matching TMultiVecId.
static_assert(std::is_convertible_v<core::VecCoordId, core::MultiVecCoordId>);
static_assert(std::is_convertible_v<core::VecCoordId, core::ConstMultiVecCoordId>);

// Write access converts implicitly to read access, never the other way round.
static_assert(std::is_convertible_v<core::MultiVecCoordId, core::ConstMultiVecCoordId>);
static_assert(std::is_convertible_v<core::MultiVecDerivId, core::ConstMultiVecDerivId>);
static_assert(std::is_convertible_v<core::MultiMatrixDerivId, core::ConstMultiMatrixDerivId>);

// A specific vtype widens implicitly to the generic V_ALL one.
static_assert(std::is_convertible_v<core::MultiVecCoordId, core::MultiVecId>);
static_assert(std::is_convertible_v<core::MultiVecCoordId, core::ConstMultiVecId>);
static_assert(std::is_convertible_v<core::MultiVecId, core::ConstMultiVecId>);

// Narrowing V_ALL back to a specific vtype stays explicit: the caller has to
// have checked the type first.
static_assert(!std::is_convertible_v<core::MultiVecId, core::MultiVecCoordId>);
static_assert(!std::is_convertible_v<core::ConstMultiVecId, core::ConstMultiVecCoordId>);
static_assert(std::is_constructible_v<core::MultiVecCoordId, core::MultiVecId>);
static_assert(std::is_constructible_v<core::ConstMultiVecCoordId, core::ConstMultiVecId>);

// Unrelated vtypes never interconvert.
static_assert(!std::is_constructible_v<core::MultiVecCoordId, core::MultiVecDerivId>);
static_assert(!std::is_constructible_v<core::MultiVecDerivId, core::MultiMatrixDerivId>);
static_assert(!std::is_assignable_v<core::MultiVecCoordId&, core::MultiVecDerivId>);

// Assignment mirrors construction, including the V_ALL -> specific direction.
static_assert(std::is_assignable_v<core::MultiVecCoordId&, core::VecCoordId>);
static_assert(std::is_assignable_v<core::ConstMultiVecCoordId&, core::MultiVecCoordId>);
static_assert(std::is_assignable_v<core::MultiVecId&, core::MultiVecCoordId>);
static_assert(std::is_assignable_v<core::MultiVecCoordId&, core::MultiVecId>);
static_assert(std::is_assignable_v<core::ConstMultiVecCoordId&, core::ConstMultiVecId>);

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
}

TEST(MultiVecIdTest, GetNameFormatAndStreaming)
{
    MockBaseState state("state1");

    core::MultiVecCoordId coord(core::vec_id::write_access::position);
    coord.setId(&state, core::vec_id::write_access::freePosition);

    // Entries whose vtype matches the default id are printed by index.
    EXPECT_EQ(coord.getName(), "{position(V_COORD)[*],3[state1]}");

    std::ostringstream ss;
    ss << coord;
    EXPECT_EQ(ss.str(), "{position(V_COORD)[*],3[state1]}");

    // Entries of a different vtype are printed by full name instead.
    core::MultiVecId generic(core::vec_id::write_access::position);
    generic.setId(&state, core::vec_id::write_access::velocity);
    EXPECT_EQ(generic.getName(), "{position(V_COORD)[*],velocity(V_DERIV)[state1]}");

    std::ostringstream genericStream;
    genericStream << generic;
    EXPECT_EQ(genericStream.str(), "{position(V_COORD)[*],velocity(V_DERIV)[state1]}");
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

// ============================================================================
// 3. Id map propagation through conversions
// ============================================================================

/// Converting a TMultiVecId must carry the per-state id map over, and must do
/// so without copying it. The no-copy part is a documented design contract:
/// BaseVecId in VecId.h holds all the data precisely so that TVecId templates
/// are layout-compatible and the shared_ptr can be shared instead of the map
/// being duplicated -- see the note on BaseVecId, which calls out passing a
/// stored TMultiVecId<!V_ALL, V_WRITE> to a const TMultiVecId<V_ALL, V_READ>&
/// as the operation this buys. That path runs several times per solver
/// iteration, so an O(n) copy there is a real regression, not a detail.
TEST(MultiVecIdTest, ConversionCarriesAndSharesIdMap)
{
    MockBaseState state("state");

    core::MultiVecCoordId src(core::vec_id::write_access::position);
    src.setId(&state, core::vec_id::write_access::freePosition);

    // Same vtype, write -> read.
    core::ConstMultiVecCoordId sameType(src);
    EXPECT_TRUE(sameType.hasIdMap());
    EXPECT_EQ(sameType.getDefaultId(), core::vec_id::read_access::position);
    EXPECT_EQ(sameType.getId(&state), core::vec_id::read_access::freePosition);
    EXPECT_EQ(static_cast<const void*>(&src.getIdMap()),
              static_cast<const void*>(&sameType.getIdMap()));

    // Specific vtype -> V_ALL.
    core::ConstMultiVecId generic(src);
    EXPECT_TRUE(generic.hasIdMap());
    EXPECT_EQ(generic.getDefaultId(), core::vec_id::read_access::position);
    EXPECT_EQ(generic.getId(&state), core::vec_id::read_access::freePosition);
    EXPECT_EQ(static_cast<const void*>(&src.getIdMap()),
              static_cast<const void*>(&generic.getIdMap()));
}

// ============================================================================
// 4. Assignment
// ============================================================================

/// operator= and assign() are deliberately not the same operation, and callers
/// rely on the difference: MechanicalParams and ConstraintParams use assign()
/// for the TVecId overload of their setters and operator= for the multi-vec
/// overload. Assigning a TVecId replaces the default id only; assign() also
/// drops the per-state overrides.
TEST(MultiVecIdTest, AssignmentFromVecIdKeepsIdMap)
{
    MockBaseState state("state");

    core::MultiVecCoordId v(core::vec_id::write_access::position);
    v.setId(&state, core::vec_id::write_access::freePosition);

    v = core::vec_id::write_access::restPosition;
    EXPECT_TRUE(v.hasIdMap());
    EXPECT_EQ(v.getId(&state), core::vec_id::write_access::freePosition);
    EXPECT_EQ(v.getDefaultId(), core::vec_id::write_access::restPosition);

    v.assign(core::vec_id::write_access::restPosition);
    EXPECT_FALSE(v.hasIdMap());
    EXPECT_EQ(v.getId(&state), core::vec_id::write_access::restPosition);
}

/// Narrowing V_ALL to a specific vtype is explicit for construction but
/// available as a plain assignment. Every id involved here is a V_COORD one:
/// the conversion asserts on the vtype of each map entry, so a map holding a
/// V_DERIV id would abort in a debug build rather than fail an expectation.
TEST(MultiVecIdTest, AssignmentFromGenericToSpecific)
{
    MockBaseState state("state");

    core::MultiVecId all(core::vec_id::write_access::position);
    all.setId(&state, core::vec_id::write_access::freePosition);

    core::MultiVecCoordId coord;
    coord = all;
    EXPECT_TRUE(coord.hasIdMap());
    EXPECT_EQ(coord.getDefaultId(), core::vec_id::write_access::position);
    EXPECT_EQ(coord.getId(&state), core::vec_id::write_access::freePosition);

    core::ConstMultiVecId constAll(core::vec_id::read_access::position);
    core::ConstMultiVecCoordId constCoord;
    constCoord = constAll;
    EXPECT_EQ(constCoord.getDefaultId(), core::vec_id::read_access::position);
}

// ============================================================================
// 5. Copy-on-write
// ============================================================================

/// writeIdMap() clones the map as soon as it is shared, so that mutating one
/// multi-vec id never reaches through to another. Every setId() elsewhere in
/// this file runs at use_count() == 1 and therefore never reaches that branch.
TEST(MultiVecIdTest, WritingToASharedIdMapClonesIt)
{
    MockBaseState state("state");

    core::MultiVecCoordId a(core::vec_id::write_access::position);
    a.setId(&state, core::vec_id::write_access::freePosition);

    core::MultiVecCoordId b(a);
    EXPECT_EQ(static_cast<const void*>(&a.getIdMap()),
              static_cast<const void*>(&b.getIdMap()));

    b.setId(&state, core::vec_id::write_access::restPosition);
    EXPECT_EQ(a.getId(&state), core::vec_id::write_access::freePosition);
    EXPECT_EQ(b.getId(&state), core::vec_id::write_access::restPosition);
    EXPECT_NE(static_cast<const void*>(&a.getIdMap()),
              static_cast<const void*>(&b.getIdMap()));

    // Same on the V_ALL specialisation.
    core::MultiVecId genericA(core::vec_id::write_access::position);
    genericA.setId(&state, core::vec_id::write_access::freePosition);

    core::MultiVecId genericB(genericA);
    genericB.setId(&state, core::vec_id::write_access::restPosition);
    EXPECT_EQ(genericA.getId(&state), core::vec_id::write_access::freePosition);
    EXPECT_EQ(genericB.getId(&state), core::vec_id::write_access::restPosition);
}

template<class DataTypes>
class MockState : public core::State<DataTypes>
{
public:
    DataVecCoord_t<DataTypes> coordData;
    DataVecDeriv_t<DataTypes> derivData;
    DataMatrixDeriv_t<DataTypes> matDerivData;

    MockState(const std::string& name = "mockState")
    {
        this->setName(name);
    }

    sofa::Size getSize() const override { return 0; }
    void resize(sofa::Size) override {}
    core::objectmodel::BaseData* baseWrite(core::VecId) override { return nullptr; }
    const core::objectmodel::BaseData* baseRead(core::ConstVecId) const override { return nullptr; }

    const DataVecCoord_t<DataTypes>* read(core::ConstVecCoordId) const override { return &coordData; }
    DataVecCoord_t<DataTypes>* write(core::VecCoordId) override { return &coordData; }

    const DataVecDeriv_t<DataTypes>* read(core::ConstVecDerivId) const override { return &derivData; }
    DataVecDeriv_t<DataTypes>* write(core::VecDerivId) override { return &derivData; }

    const DataMatrixDeriv_t<DataTypes>* read(core::ConstMatrixDerivId) const override { return &matDerivData; }
    DataMatrixDeriv_t<DataTypes>* write(core::MatrixDerivId) override { return &matDerivData; }
};

// ============================================================================
// 6. Subscript operator (operator[]) and StateVecAccessor tests
// ============================================================================

TEST(MultiVecIdTest, SubscriptOperatorAndStateVecAccessorCoord)
{
    MockState<defaulttype::Vec1Types> state1("state1");
    MockState<defaulttype::Rigid2Types> state2("state2");
    const MockState<defaulttype::Vec1Types>* constState1 = &state1;

    // V_COORD, V_WRITE
    core::MultiVecCoordId writeMulti(core::vec_id::write_access::position);
    writeMulti.setId(&state1, core::vec_id::write_access::freePosition);

    auto writeAcc1 = writeMulti[&state1];
    static_assert(std::is_same_v<decltype(writeAcc1), core::StateVecAccessor<defaulttype::Vec1Types, core::V_COORD, core::V_WRITE>>);
    core::VecCoordId writeId1 = writeAcc1;
    EXPECT_EQ(writeId1, core::vec_id::write_access::freePosition);
    EXPECT_EQ(writeAcc1.read(), &state1.coordData);
    EXPECT_EQ(writeAcc1.write(), &state1.coordData);

    auto writeAcc2 = writeMulti[&state2];
    static_assert(std::is_same_v<decltype(writeAcc2), core::StateVecAccessor<defaulttype::Rigid2Types, core::V_COORD, core::V_WRITE>>);
    core::VecCoordId writeId2 = writeAcc2;
    EXPECT_EQ(writeId2, core::vec_id::write_access::position);
    EXPECT_EQ(writeAcc2.read(), &state2.coordData);
    EXPECT_EQ(writeAcc2.write(), &state2.coordData);

    // Const state access on write multi-vec yields read-only accessor
    auto readAccFromConstState = writeMulti[constState1];
    static_assert(std::is_same_v<decltype(readAccFromConstState), core::StateVecAccessor<defaulttype::Vec1Types, core::V_COORD, core::V_READ>>);
    core::ConstVecCoordId readIdFromConst = readAccFromConstState;
    EXPECT_EQ(readIdFromConst, core::vec_id::read_access::freePosition);
    EXPECT_EQ(readAccFromConstState.read(), &state1.coordData);

    // V_COORD, V_READ
    core::ConstMultiVecCoordId readMulti(core::vec_id::read_access::position);
    readMulti.setId(&state1, core::vec_id::read_access::freePosition);

    auto readAcc1 = readMulti[&state1];
    static_assert(std::is_same_v<decltype(readAcc1), core::StateVecAccessor<defaulttype::Vec1Types, core::V_COORD, core::V_READ>>);
    core::ConstVecCoordId readId1 = readAcc1;
    EXPECT_EQ(readId1, core::vec_id::read_access::freePosition);
    EXPECT_EQ(readAcc1.read(), &state1.coordData);

    auto readAccFromConst = readMulti[constState1];
    static_assert(std::is_same_v<decltype(readAccFromConst), core::StateVecAccessor<defaulttype::Vec1Types, core::V_COORD, core::V_READ>>);
    EXPECT_EQ(readAccFromConst.read(), &state1.coordData);
}

TEST(MultiVecIdTest, SubscriptOperatorAndStateVecAccessorDeriv)
{
    MockState<defaulttype::Vec1Types> state1("state1");
    const MockState<defaulttype::Vec1Types>* constState1 = &state1;

    // V_DERIV, V_WRITE
    core::MultiVecDerivId writeMulti(core::vec_id::write_access::velocity);
    writeMulti.setId(&state1, core::vec_id::write_access::force);

    auto writeAcc = writeMulti[&state1];
    static_assert(std::is_same_v<decltype(writeAcc), core::StateVecAccessor<defaulttype::Vec1Types, core::V_DERIV, core::V_WRITE>>);
    core::VecDerivId writeId = writeAcc;
    EXPECT_EQ(writeId, core::vec_id::write_access::force);
    EXPECT_EQ(writeAcc.read(), &state1.derivData);
    EXPECT_EQ(writeAcc.write(), &state1.derivData);

    // V_DERIV, V_READ
    core::ConstMultiVecDerivId readMulti(core::vec_id::read_access::velocity);
    readMulti.setId(&state1, core::vec_id::read_access::force);

    auto readAcc = readMulti[constState1];
    static_assert(std::is_same_v<decltype(readAcc), core::StateVecAccessor<defaulttype::Vec1Types, core::V_DERIV, core::V_READ>>);
    core::ConstVecDerivId readId = readAcc;
    EXPECT_EQ(readId, core::vec_id::read_access::force);
    EXPECT_EQ(readAcc.read(), &state1.derivData);
}

TEST(MultiVecIdTest, SubscriptOperatorAndStateVecAccessorMatDeriv)
{
    MockState<defaulttype::Vec1Types> state1("state1");
    const MockState<defaulttype::Vec1Types>* constState1 = &state1;

    // V_MATDERIV, V_WRITE
    core::MultiMatrixDerivId writeMulti(core::vec_id::write_access::constraintJacobian);
    writeMulti.setId(&state1, core::vec_id::write_access::mappingJacobian);

    auto writeAcc = writeMulti[&state1];
    static_assert(std::is_same_v<decltype(writeAcc), core::StateVecAccessor<defaulttype::Vec1Types, core::V_MATDERIV, core::V_WRITE>>);
    core::MatrixDerivId writeId = writeAcc;
    EXPECT_EQ(writeId, core::vec_id::write_access::mappingJacobian);
    EXPECT_EQ(writeAcc.read(), &state1.matDerivData);
    EXPECT_EQ(writeAcc.write(), &state1.matDerivData);

    // V_MATDERIV, V_READ
    core::ConstMultiMatrixDerivId readMulti(core::vec_id::read_access::constraintJacobian);
    readMulti.setId(&state1, core::vec_id::read_access::mappingJacobian);

    auto readAcc = readMulti[constState1];
    static_assert(std::is_same_v<decltype(readAcc), core::StateVecAccessor<defaulttype::Vec1Types, core::V_MATDERIV, core::V_READ>>);
    core::ConstMatrixDerivId readId = readAcc;
    EXPECT_EQ(readId, core::vec_id::read_access::mappingJacobian);
    EXPECT_EQ(readAcc.read(), &state1.matDerivData);
}

TEST(MultiVecIdTest, SubscriptOperatorAndStateVecAccessorAll)
{
    MockState<defaulttype::Vec1Types> state1("state1");
    const MockState<defaulttype::Vec1Types>* constState1 = &state1;

    // V_ALL, V_WRITE
    core::MultiVecId writeMulti(core::vec_id::write_access::position);
    writeMulti.setId(&state1, core::vec_id::write_access::velocity);

    auto writeAcc = writeMulti[&state1];
    static_assert(std::is_same_v<decltype(writeAcc), core::StateVecAccessor<defaulttype::Vec1Types, core::V_ALL, core::V_WRITE>>);
    core::VecId writeId = writeAcc;
    EXPECT_EQ(writeId, core::vec_id::write_access::velocity);

    // V_ALL, V_READ
    core::ConstMultiVecId readMulti(core::vec_id::read_access::position);
    readMulti.setId(&state1, core::vec_id::read_access::velocity);

    auto readAcc = readMulti[constState1];
    static_assert(std::is_same_v<decltype(readAcc), core::StateVecAccessor<defaulttype::Vec1Types, core::V_ALL, core::V_READ>>);
    core::ConstVecId readId = readAcc;
    EXPECT_EQ(readId, core::vec_id::read_access::velocity);
}


} // namespace sofa
