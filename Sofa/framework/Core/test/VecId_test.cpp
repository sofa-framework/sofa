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
    static constexpr auto position = sofa::core::vec_id::read_access::position;
    EXPECT_EQ(position.getName(), "position(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::position.getName(), "position(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::position.getName(), "position(V_COORD)");

    static constexpr auto restPosition = sofa::core::vec_id::read_access::restPosition;
    EXPECT_EQ(restPosition.getName(), "restPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::restPosition.getName(), "restPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::restPosition.getName(), "restPosition(V_COORD)");

    static constexpr auto freePosition = sofa::core::vec_id::read_access::freePosition;
    EXPECT_EQ(freePosition.getName(), "freePosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::freePosition.getName(), "freePosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::freePosition.getName(), "freePosition(V_COORD)");

    static constexpr auto resetPosition = sofa::core::vec_id::read_access::resetPosition;
    EXPECT_EQ(resetPosition.getName(), "resetPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::read_access::resetPosition.getName(), "resetPosition(V_COORD)");
    EXPECT_EQ(sofa::core::vec_id::write_access::resetPosition.getName(), "resetPosition(V_COORD)");


    static constexpr auto velocity = sofa::core::vec_id::read_access::velocity;
    EXPECT_EQ(velocity.getName(), "velocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::velocity.getName(), "velocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::velocity.getName(), "velocity(V_DERIV)");

    static constexpr auto resetVelocity = sofa::core::vec_id::read_access::resetVelocity;
    EXPECT_EQ(resetVelocity.getName(), "resetVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::resetVelocity.getName(), "resetVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::resetVelocity.getName(), "resetVelocity(V_DERIV)");

    static constexpr auto freeVelocity = sofa::core::vec_id::read_access::freeVelocity;
    EXPECT_EQ(freeVelocity.getName(), "freeVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::freeVelocity.getName(), "freeVelocity(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::freeVelocity.getName(), "freeVelocity(V_DERIV)");

    static constexpr auto normal = sofa::core::vec_id::read_access::normal;
    EXPECT_EQ(normal.getName(), "normal(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::normal.getName(), "normal(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::normal.getName(), "normal(V_DERIV)");

    static constexpr auto force = sofa::core::vec_id::read_access::force;
    EXPECT_EQ(force.getName(), "force(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::force.getName(), "force(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::force.getName(), "force(V_DERIV)");

    static constexpr auto externalForce = sofa::core::vec_id::read_access::externalForce;
    EXPECT_EQ(externalForce.getName(), "externalForce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::externalForce.getName(), "externalForce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::externalForce.getName(), "externalForce(V_DERIV)");

    static constexpr auto dx = sofa::core::vec_id::read_access::dx;
    EXPECT_EQ(dx.getName(), "dx(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::dx.getName(), "dx(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::dx.getName(), "dx(V_DERIV)");

    static constexpr auto dforce = sofa::core::vec_id::read_access::dforce;
    EXPECT_EQ(dforce.getName(), "dforce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::read_access::dforce.getName(), "dforce(V_DERIV)");
    EXPECT_EQ(sofa::core::vec_id::write_access::dforce.getName(), "dforce(V_DERIV)");

    const std::string s = Sofa.Component.Collision;
    EXPECT_EQ(s, std::string("Sofa.Component.Collision"));

}

TEST(TVecId, ConstructorsAndAccessors)
{
    using namespace sofa::core;

    // Specific types - default constructors
    constexpr ConstVecCoordId defaultCoordRead;
    EXPECT_EQ(defaultCoordRead.getType(), VecType::V_COORD);
    EXPECT_EQ(defaultCoordRead.getIndex(), 0u);

    constexpr VecCoordId defaultCoordWrite;
    EXPECT_EQ(defaultCoordWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(defaultCoordWrite.getIndex(), 0u);

    constexpr ConstVecDerivId defaultDerivRead;
    EXPECT_EQ(defaultDerivRead.getType(), VecType::V_DERIV);
    EXPECT_EQ(defaultDerivRead.getIndex(), 0u);

    constexpr VecDerivId defaultDerivWrite;
    EXPECT_EQ(defaultDerivWrite.getType(), VecType::V_DERIV);
    EXPECT_EQ(defaultDerivWrite.getIndex(), 0u);

    constexpr ConstMatrixDerivId defaultMatDerivRead;
    EXPECT_EQ(defaultMatDerivRead.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(defaultMatDerivRead.getIndex(), 0u);

    constexpr MatrixDerivId defaultMatDerivWrite;
    EXPECT_EQ(defaultMatDerivWrite.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(defaultMatDerivWrite.getIndex(), 0u);

    // Specific types - explicit index constructors
    constexpr ConstVecCoordId coordRead(5);
    EXPECT_EQ(coordRead.getType(), VecType::V_COORD);
    EXPECT_EQ(coordRead.getIndex(), 5u);

    constexpr VecCoordId coordWrite(10);
    EXPECT_EQ(coordWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(coordWrite.getIndex(), 10u);

    constexpr ConstVecDerivId derivRead(7);
    EXPECT_EQ(derivRead.getType(), VecType::V_DERIV);
    EXPECT_EQ(derivRead.getIndex(), 7u);

    constexpr VecDerivId derivWrite(12);
    EXPECT_EQ(derivWrite.getType(), VecType::V_DERIV);
    EXPECT_EQ(derivWrite.getIndex(), 12u);

    constexpr ConstMatrixDerivId matDerivRead(3);
    EXPECT_EQ(matDerivRead.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(matDerivRead.getIndex(), 3u);

    constexpr MatrixDerivId matDerivWrite(8);
    EXPECT_EQ(matDerivWrite.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(matDerivWrite.getIndex(), 8u);

    // Generic TVecId<VecType::V_ALL, ...> constructors
    constexpr ConstVecId defaultGenRead;
    EXPECT_EQ(defaultGenRead.getType(), VecType::V_ALL);
    EXPECT_EQ(defaultGenRead.getIndex(), 0u);

    constexpr VecId defaultGenWrite;
    EXPECT_EQ(defaultGenWrite.getType(), VecType::V_ALL);
    EXPECT_EQ(defaultGenWrite.getIndex(), 0u);

    constexpr ConstVecId genReadCoord(VecType::V_COORD, 3);
    EXPECT_EQ(genReadCoord.getType(), VecType::V_COORD);
    EXPECT_EQ(genReadCoord.getIndex(), 3u);

    constexpr VecId genWriteDeriv(VecType::V_DERIV, 9);
    EXPECT_EQ(genWriteDeriv.getType(), VecType::V_DERIV);
    EXPECT_EQ(genWriteDeriv.getIndex(), 9u);
}

TEST(TVecId, NullAndIsNull)
{
    using namespace sofa::core;

    // Specific types null() and isNull()
    constexpr auto nullCoordRead = ConstVecCoordId::null();
    EXPECT_TRUE(nullCoordRead.isNull());
    EXPECT_EQ(nullCoordRead.getIndex(), 0u);
    EXPECT_FALSE(ConstVecCoordId(1).isNull());

    constexpr auto nullCoordWrite = VecCoordId::null();
    EXPECT_TRUE(nullCoordWrite.isNull());
    EXPECT_FALSE(VecCoordId(2).isNull());

    constexpr auto nullDerivRead = ConstVecDerivId::null();
    EXPECT_TRUE(nullDerivRead.isNull());
    EXPECT_FALSE(ConstVecDerivId(1).isNull());

    constexpr auto nullDerivWrite = VecDerivId::null();
    EXPECT_TRUE(nullDerivWrite.isNull());
    EXPECT_FALSE(VecDerivId(2).isNull());

    constexpr auto nullMatDerivRead = ConstMatrixDerivId::null();
    EXPECT_TRUE(nullMatDerivRead.isNull());
    EXPECT_FALSE(ConstMatrixDerivId(1).isNull());

    constexpr auto nullMatDerivWrite = MatrixDerivId::null();
    EXPECT_TRUE(nullMatDerivWrite.isNull());
    EXPECT_FALSE(MatrixDerivId(2).isNull());

    // Generic VecType::V_ALL null() and isNull()
    constexpr auto nullGenRead = ConstVecId::null();
    EXPECT_TRUE(nullGenRead.isNull());
    EXPECT_FALSE(ConstVecId(VecType::V_COORD, 1).isNull());

    constexpr auto nullGenWrite = VecId::null();
    EXPECT_TRUE(nullGenWrite.isNull());
    EXPECT_FALSE(VecId(VecType::V_DERIV, 2).isNull());

    // Access: write-access ids convert implicitly to read-access ids.
    static_assert(std::is_convertible_v<VecCoordId, ConstVecCoordId>);
    static_assert(std::is_convertible_v<VecDerivId, ConstVecDerivId>);
    static_assert(std::is_convertible_v<MatrixDerivId, ConstMatrixDerivId>);

    // A specific id converts implicitly to the generic V_ALL id.
    static_assert(std::is_convertible_v<VecCoordId, VecId>);
    static_assert(std::is_convertible_v<VecCoordId, ConstVecId>);

    // The reverse narrowing must stay explicit: constructible, never implicit.
    static_assert(!std::is_convertible_v<VecId, VecCoordId>);
    static_assert(std::is_constructible_v<VecCoordId, VecId>);

    // Unrelated specific types never interconvert.
    static_assert(!std::is_constructible_v<VecCoordId, VecDerivId>);
    static_assert(!std::is_assignable_v<VecCoordId&, VecDerivId>);

    // Assignment mirrors construction, including the V_ALL -> specific direction.
    static_assert(std::is_assignable_v<VecCoordId&, VecId>);
    static_assert(std::is_assignable_v<VecId&, VecCoordId>);
}

TEST(TVecId, DynamicIndex)
{
    using namespace sofa::core;

    EXPECT_EQ((TStandardVec<VecType::V_COORD, V_READ>::V_FIRST_DYNAMIC_INDEX), static_cast<uint8_t>(CoordState::DYNAMIC_INDEX));
    EXPECT_EQ((TStandardVec<VecType::V_COORD, V_WRITE>::V_FIRST_DYNAMIC_INDEX), static_cast<uint8_t>(CoordState::DYNAMIC_INDEX));

    EXPECT_EQ((TStandardVec<VecType::V_DERIV, V_READ>::V_FIRST_DYNAMIC_INDEX), static_cast<uint8_t>(DerivState::DYNAMIC_INDEX));
    EXPECT_EQ((TStandardVec<VecType::V_DERIV, V_WRITE>::V_FIRST_DYNAMIC_INDEX), static_cast<uint8_t>(DerivState::DYNAMIC_INDEX));

    EXPECT_EQ((TStandardVec<VecType::V_MATDERIV, V_READ>::V_FIRST_DYNAMIC_INDEX), static_cast<uint8_t>(MatrixDerivState::DYNAMIC_INDEX));
    EXPECT_EQ((TStandardVec<VecType::V_MATDERIV, V_WRITE>::V_FIRST_DYNAMIC_INDEX), static_cast<uint8_t>(MatrixDerivState::DYNAMIC_INDEX));

    EXPECT_EQ((TStandardVec<VecType::V_ALL, V_READ>::getFirstDynamicIndex(VecType::V_COORD)), static_cast<uint8_t>(CoordState::DYNAMIC_INDEX));
    EXPECT_EQ((TStandardVec<VecType::V_ALL, V_READ>::getFirstDynamicIndex(VecType::V_DERIV)), static_cast<uint8_t>(DerivState::DYNAMIC_INDEX));
    EXPECT_EQ((TStandardVec<VecType::V_ALL, V_READ>::getFirstDynamicIndex(VecType::V_MATDERIV)), static_cast<uint8_t>(MatrixDerivState::DYNAMIC_INDEX));
    EXPECT_EQ((TStandardVec<VecType::V_ALL, V_READ>::getFirstDynamicIndex(VecType::V_ALL)), 0u);
}

TEST(TVecId, CopyConstructorsAllCombinations)
{
    using namespace sofa::core;

    // 1. Same specific type & same access
    constexpr VecCoordId coordWrite(4);
    constexpr VecCoordId coordWriteCopy(coordWrite);
    EXPECT_EQ(coordWriteCopy.getIndex(), 4u);
    EXPECT_EQ(coordWriteCopy.getType(), VecType::V_COORD);

    constexpr ConstVecCoordId coordRead(3);
    constexpr ConstVecCoordId coordReadCopy(coordRead);
    EXPECT_EQ(coordReadCopy.getIndex(), 3u);

    // 2. Write-to-Read conversion (specific type)
    constexpr ConstVecCoordId coordReadFromWrite(coordWrite);
    EXPECT_EQ(coordReadFromWrite.getIndex(), 4u);
    EXPECT_EQ(coordReadFromWrite.getType(), VecType::V_COORD);

    constexpr VecDerivId derivWrite(6);
    constexpr ConstVecDerivId derivReadFromWrite(derivWrite);
    EXPECT_EQ(derivReadFromWrite.getIndex(), 6u);
    EXPECT_EQ(derivReadFromWrite.getType(), VecType::V_DERIV);

    constexpr MatrixDerivId matDerivWrite(7);
    constexpr ConstMatrixDerivId matDerivReadFromWrite(matDerivWrite);
    EXPECT_EQ(matDerivReadFromWrite.getIndex(), 7u);
    EXPECT_EQ(matDerivReadFromWrite.getType(), VecType::V_MATDERIV);

    // 3. Specific to Generic (TVecId<VecType::V_ALL, ...>)
    // From Write specific to Write generic
    constexpr VecId genWriteFromCoordWrite(coordWrite);
    EXPECT_EQ(genWriteFromCoordWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(genWriteFromCoordWrite.getIndex(), 4u);

    constexpr VecId genWriteFromDerivWrite(derivWrite);
    EXPECT_EQ(genWriteFromDerivWrite.getType(), VecType::V_DERIV);
    EXPECT_EQ(genWriteFromDerivWrite.getIndex(), 6u);

    constexpr VecId genWriteFromMatDerivWrite(matDerivWrite);
    EXPECT_EQ(genWriteFromMatDerivWrite.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(genWriteFromMatDerivWrite.getIndex(), 7u);

    // From Write specific to Read generic
    constexpr ConstVecId genReadFromCoordWrite(coordWrite);
    EXPECT_EQ(genReadFromCoordWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(genReadFromCoordWrite.getIndex(), 4u);

    constexpr ConstVecId genReadFromDerivWrite(derivWrite);
    EXPECT_EQ(genReadFromDerivWrite.getType(), VecType::V_DERIV);
    EXPECT_EQ(genReadFromDerivWrite.getIndex(), 6u);

    constexpr ConstVecId genReadFromMatDerivWrite(matDerivWrite);
    EXPECT_EQ(genReadFromMatDerivWrite.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(genReadFromMatDerivWrite.getIndex(), 7u);

    // From Read specific to Read generic
    constexpr ConstVecId genReadFromCoordRead(coordRead);
    EXPECT_EQ(genReadFromCoordRead.getType(), VecType::V_COORD);
    EXPECT_EQ(genReadFromCoordRead.getIndex(), 3u);

    // 4. Generic to Specific (explicit constructor)
    constexpr VecId genCoordWrite(VecType::V_COORD, 8);
    constexpr VecCoordId coordWriteFromGenWrite(genCoordWrite);
    EXPECT_EQ(coordWriteFromGenWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(coordWriteFromGenWrite.getIndex(), 8u);

    constexpr ConstVecCoordId coordReadFromGenWrite(genCoordWrite);
    EXPECT_EQ(coordReadFromGenWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(coordReadFromGenWrite.getIndex(), 8u);

    constexpr ConstVecId genCoordRead(VecType::V_COORD, 9);
    constexpr ConstVecCoordId coordReadFromGenRead(genCoordRead);
    EXPECT_EQ(coordReadFromGenRead.getType(), VecType::V_COORD);
    EXPECT_EQ(coordReadFromGenRead.getIndex(), 9u);

    constexpr VecId genDerivWrite(VecType::V_DERIV, 11);
    constexpr VecDerivId derivWriteFromGenWrite(genDerivWrite);
    EXPECT_EQ(derivWriteFromGenWrite.getType(), VecType::V_DERIV);
    EXPECT_EQ(derivWriteFromGenWrite.getIndex(), 11u);

    constexpr MatrixDerivId matDerivWriteFromGenWrite(VecId(VecType::V_MATDERIV, 13));
    EXPECT_EQ(matDerivWriteFromGenWrite.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(matDerivWriteFromGenWrite.getIndex(), 13u);

    // 5. Generic to Generic
    constexpr VecId genWrite(VecType::V_COORD, 15);
    constexpr VecId genWriteCopy(genWrite);
    EXPECT_EQ(genWriteCopy.getType(), VecType::V_COORD);
    EXPECT_EQ(genWriteCopy.getIndex(), 15u);

    constexpr ConstVecId genReadFromGenWrite(genWrite);
    EXPECT_EQ(genReadFromGenWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(genReadFromGenWrite.getIndex(), 15u);

    constexpr ConstVecId genRead(VecType::V_DERIV, 16);
    constexpr ConstVecId genReadCopy(genRead);
    EXPECT_EQ(genReadCopy.getType(), VecType::V_DERIV);
    EXPECT_EQ(genReadCopy.getIndex(), 16u);
}

TEST(TVecId, CopyAssignmentAllCombinations)
{
    using namespace sofa::core;

    // 1. Same specific type & access
    VecCoordId coordWrite1(1);
    VecCoordId coordWrite2(2);
    coordWrite1 = coordWrite2;
    EXPECT_EQ(coordWrite1.getIndex(), 2u);

    // 2. Write-to-Read assignment (specific type)
    ConstVecCoordId coordRead(0);
    coordRead = coordWrite2;
    EXPECT_EQ(coordRead.getIndex(), 2u);

    VecDerivId derivWrite(4);
    ConstVecDerivId derivRead(0);
    derivRead = derivWrite;
    EXPECT_EQ(derivRead.getIndex(), 4u);

    MatrixDerivId matDerivWrite(5);
    ConstMatrixDerivId matDerivRead(0);
    matDerivRead = matDerivWrite;
    EXPECT_EQ(matDerivRead.getIndex(), 5u);

    // 3. Specific to Generic (TVecId<VecType::V_ALL, ...>)
    VecId genWrite;
    genWrite = coordWrite2;
    EXPECT_EQ(genWrite.getType(), VecType::V_COORD);
    EXPECT_EQ(genWrite.getIndex(), 2u);

    genWrite = derivWrite;
    EXPECT_EQ(genWrite.getType(), VecType::V_DERIV);
    EXPECT_EQ(genWrite.getIndex(), 4u);

    genWrite = matDerivWrite;
    EXPECT_EQ(genWrite.getType(), VecType::V_MATDERIV);
    EXPECT_EQ(genWrite.getIndex(), 5u);

    ConstVecId genRead;
    genRead = coordWrite2; // Write to Read
    EXPECT_EQ(genRead.getType(), VecType::V_COORD);
    EXPECT_EQ(genRead.getIndex(), 2u);

    genRead = coordRead; // Read to Read
    EXPECT_EQ(genRead.getType(), VecType::V_COORD);
    EXPECT_EQ(genRead.getIndex(), 2u);

    // 4. Generic to Specific
    VecCoordId coordTargetWrite;
    coordTargetWrite = VecId(VecType::V_COORD, 20);
    EXPECT_EQ(coordTargetWrite.getIndex(), 20u);
    EXPECT_EQ(coordTargetWrite.getType(), VecType::V_COORD);

    ConstVecCoordId coordTargetRead;
    coordTargetRead = VecId(VecType::V_COORD, 21); // Write generic to Read specific
    EXPECT_EQ(coordTargetRead.getIndex(), 21u);
    EXPECT_EQ(coordTargetRead.getType(), VecType::V_COORD);

    coordTargetRead = ConstVecId(VecType::V_COORD, 22); // Read generic to Read specific
    EXPECT_EQ(coordTargetRead.getIndex(), 22u);
    EXPECT_EQ(coordTargetRead.getType(), VecType::V_COORD);

    VecDerivId derivTargetWrite;
    derivTargetWrite = VecId(VecType::V_DERIV, 23);
    EXPECT_EQ(derivTargetWrite.getIndex(), 23u);
    EXPECT_EQ(derivTargetWrite.getType(), VecType::V_DERIV);

    MatrixDerivId matDerivTargetWrite;
    matDerivTargetWrite = VecId(VecType::V_MATDERIV, 24);
    EXPECT_EQ(matDerivTargetWrite.getIndex(), 24u);
    EXPECT_EQ(matDerivTargetWrite.getType(), VecType::V_MATDERIV);

    // 5. Generic to Generic
    VecId genA(VecType::V_COORD, 30);
    VecId genB;
    genB = genA;
    EXPECT_EQ(genB.getType(), VecType::V_COORD);
    EXPECT_EQ(genB.getIndex(), 30u);

    ConstVecId genC;
    genC = genA; // Write generic to Read generic
    EXPECT_EQ(genC.getType(), VecType::V_COORD);
    EXPECT_EQ(genC.getIndex(), 30u);
}

TEST(TVecId, EqualityAndInequalityComparisons)
{
    using namespace sofa::core;

    constexpr VecCoordId coordWrite1(5);
    constexpr ConstVecCoordId coordRead1(5);
    constexpr VecCoordId coordWrite2(6);
    constexpr VecDerivId derivWrite1(5);
    constexpr ConstVecDerivId derivRead1(5);
    constexpr MatrixDerivId matDerivWrite1(5);

    // Cross access comparison (same type & index)
    EXPECT_TRUE(coordWrite1 == coordRead1);
    EXPECT_FALSE(coordWrite1 != coordRead1);
    EXPECT_TRUE(coordRead1 == coordWrite1);
    EXPECT_FALSE(coordRead1 != coordWrite1);

    // Same type, different index
    EXPECT_FALSE(coordWrite1 == coordWrite2);
    EXPECT_TRUE(coordWrite1 != coordWrite2);

    // Different type, same index
    EXPECT_FALSE(coordWrite1 == derivWrite1);
    EXPECT_TRUE(coordWrite1 != derivWrite1);
    EXPECT_FALSE(coordWrite1 == matDerivWrite1);
    EXPECT_TRUE(coordWrite1 != matDerivWrite1);
    EXPECT_FALSE(derivRead1 == matDerivWrite1);
    EXPECT_TRUE(derivRead1 != matDerivWrite1);

    // Comparison with generic VecType::V_ALL
    constexpr VecId genCoordWrite(VecType::V_COORD, 5);
    constexpr ConstVecId genCoordRead(VecType::V_COORD, 5);
    constexpr VecId genDerivWrite(VecType::V_DERIV, 5);
    constexpr VecId genCoordDifferentIndex(VecType::V_COORD, 6);

    EXPECT_TRUE(coordWrite1 == genCoordWrite);
    EXPECT_TRUE(coordRead1 == genCoordRead);
    EXPECT_TRUE(genCoordWrite == coordWrite1);
    EXPECT_FALSE(coordWrite1 != genCoordWrite);

    EXPECT_FALSE(coordWrite1 == genDerivWrite);
    EXPECT_TRUE(coordWrite1 != genDerivWrite);
    EXPECT_FALSE(coordWrite1 == genCoordDifferentIndex);
    EXPECT_TRUE(coordWrite1 != genCoordDifferentIndex);
}

TEST(TVecId, Groups)
{
    using namespace sofa::core;

    // Coord groups
    EXPECT_EQ(VecCoordId::state<CoordState::NULL_STATE>().getGroup(), "");
    EXPECT_EQ(VecCoordId::state<CoordState::POSITION>().getGroup(), "States");
    EXPECT_EQ(VecCoordId::state<CoordState::REST_POSITION>().getGroup(), "Rest States");
    EXPECT_EQ(VecCoordId::state<CoordState::FREE_POSITION>().getGroup(), "Free Motion");
    EXPECT_EQ(VecCoordId::state<CoordState::RESET_POSITION>().getGroup(), "States");
    EXPECT_EQ(VecCoordId(99).getGroup(), "");

    EXPECT_EQ(ConstVecCoordId::state<CoordState::NULL_STATE>().getGroup(), "");
    EXPECT_EQ(ConstVecCoordId::state<CoordState::POSITION>().getGroup(), "States");
    EXPECT_EQ(ConstVecCoordId::state<CoordState::REST_POSITION>().getGroup(), "Rest States");
    EXPECT_EQ(ConstVecCoordId::state<CoordState::FREE_POSITION>().getGroup(), "Free Motion");
    EXPECT_EQ(ConstVecCoordId::state<CoordState::RESET_POSITION>().getGroup(), "States");
    EXPECT_EQ(ConstVecCoordId(99).getGroup(), "");

    // Deriv groups
    EXPECT_EQ(VecDerivId::state<DerivState::NULL_STATE>().getGroup(), "");
    EXPECT_EQ(VecDerivId::state<DerivState::VELOCITY>().getGroup(), "States");
    EXPECT_EQ(VecDerivId::state<DerivState::DX>().getGroup(), "States");
    EXPECT_EQ(VecDerivId::state<DerivState::NORMAL>().getGroup(), "States");
    EXPECT_EQ(VecDerivId::state<DerivState::RESET_VELOCITY>().getGroup(), "States");
    EXPECT_EQ(VecDerivId::state<DerivState::FREE_VELOCITY>().getGroup(), "Free Motion");
    EXPECT_EQ(VecDerivId::state<DerivState::FORCE>().getGroup(), "Force");
    EXPECT_EQ(VecDerivId::state<DerivState::DFORCE>().getGroup(), "Force");
    EXPECT_EQ(VecDerivId::state<DerivState::EXTERNAL_FORCE>().getGroup(), "Force");
    EXPECT_EQ(VecDerivId(99).getGroup(), "");

    EXPECT_EQ(ConstVecDerivId::state<DerivState::NULL_STATE>().getGroup(), "");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::VELOCITY>().getGroup(), "States");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::DX>().getGroup(), "States");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::NORMAL>().getGroup(), "States");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::RESET_VELOCITY>().getGroup(), "States");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::FREE_VELOCITY>().getGroup(), "Free Motion");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::FORCE>().getGroup(), "Force");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::DFORCE>().getGroup(), "Force");
    EXPECT_EQ(ConstVecDerivId::state<DerivState::EXTERNAL_FORCE>().getGroup(), "Force");
    EXPECT_EQ(ConstVecDerivId(99).getGroup(), "");

    // Matrix Deriv groups
    EXPECT_EQ(MatrixDerivId::state<MatrixDerivState::NULL_STATE>().getGroup(), "");
    EXPECT_EQ(MatrixDerivId::state<MatrixDerivState::CONSTRAINT_JACOBIAN>().getGroup(), "Jacobian");
    EXPECT_EQ(MatrixDerivId::state<MatrixDerivState::MAPPING_JACOBIAN>().getGroup(), "Jacobian");
    EXPECT_EQ(MatrixDerivId(99).getGroup(), "");

    EXPECT_EQ(ConstMatrixDerivId::state<MatrixDerivState::NULL_STATE>().getGroup(), "");
    EXPECT_EQ(ConstMatrixDerivId::state<MatrixDerivState::CONSTRAINT_JACOBIAN>().getGroup(), "Jacobian");
    EXPECT_EQ(ConstMatrixDerivId::state<MatrixDerivState::MAPPING_JACOBIAN>().getGroup(), "Jacobian");
    EXPECT_EQ(ConstMatrixDerivId(99).getGroup(), "");

    // Generic VecType::V_ALL groups
    EXPECT_EQ(VecId(VecType::V_COORD, static_cast<unsigned int>(CoordState::POSITION)).getGroup(), "States");
    EXPECT_EQ(VecId(VecType::V_DERIV, static_cast<unsigned int>(DerivState::FORCE)).getGroup(), "Force");
    EXPECT_EQ(VecId(VecType::V_MATDERIV, static_cast<unsigned int>(MatrixDerivState::CONSTRAINT_JACOBIAN)).getGroup(), "Jacobian");

    EXPECT_EQ(ConstVecId(VecType::V_COORD, static_cast<unsigned int>(CoordState::POSITION)).getGroup(), "States");
    EXPECT_EQ(ConstVecId(VecType::V_DERIV, static_cast<unsigned int>(DerivState::FORCE)).getGroup(), "Force");
    EXPECT_EQ(ConstVecId(VecType::V_MATDERIV, static_cast<unsigned int>(MatrixDerivState::CONSTRAINT_JACOBIAN)).getGroup(), "Jacobian");

}

TEST(TVecId, DynamicAndCustomNames)
{
    using namespace sofa::core;

    EXPECT_EQ(VecCoordId::null().getName(), "null(V_COORD)");
    EXPECT_EQ(VecCoordId(10).getName(), "10(V_COORD)");

    EXPECT_EQ(VecDerivId::null().getName(), "null(V_DERIV)");
    EXPECT_EQ(VecDerivId(10).getName(), "10(V_DERIV)");

    EXPECT_EQ(MatrixDerivId::null().getName(), "null(V_MATDERIV)");
    EXPECT_EQ(MatrixDerivId::state<MatrixDerivState::CONSTRAINT_JACOBIAN>().getName(), "holonomic(V_MATDERIV)");
    EXPECT_EQ(MatrixDerivId::state<MatrixDerivState::MAPPING_JACOBIAN>().getName(), "nonHolonomic(V_MATDERIV)");
    EXPECT_EQ(MatrixDerivId(10).getName(), "10(V_MATDERIV)");

    // Generic VecType::V_ALL dynamic names
    EXPECT_EQ(VecId(VecType::V_COORD, 10).getName(), "10(V_COORD)");
    EXPECT_EQ(VecId(VecType::V_DERIV, 10).getName(), "10(V_DERIV)");
    EXPECT_EQ(VecId(VecType::V_MATDERIV, 10).getName(), "10(V_MATDERIV)");
}

TEST(TVecId, StreamOperator)
{
    using namespace sofa::core;

    auto testStream = [](const auto& vecId) {
        std::ostringstream ss;
        ss << vecId;
        EXPECT_EQ(ss.str(), vecId.getName());
    };

    testStream(VecCoordId::state<CoordState::POSITION>());
    testStream(ConstVecCoordId::state<CoordState::REST_POSITION>());
    testStream(VecDerivId::state<DerivState::VELOCITY>());
    testStream(ConstVecDerivId::state<DerivState::FORCE>());
    testStream(MatrixDerivId::state<MatrixDerivState::CONSTRAINT_JACOBIAN>());
    testStream(ConstMatrixDerivId::state<MatrixDerivState::MAPPING_JACOBIAN>());
    testStream(VecId(VecType::V_COORD, 42));
    testStream(ConstVecId(VecType::V_DERIV, 24));
}
