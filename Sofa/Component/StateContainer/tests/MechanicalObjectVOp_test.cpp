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
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/testing/NumericTest.h>

namespace sofa
{

template<class DataTypes>
struct MechanicalObjectVOpTest : public testing::BaseTest
{
    using MO = component::statecontainer::MechanicalObject<DataTypes>;

    static constexpr bool isRigid = type::isRigidType<DataTypes>;

    static constexpr Real_t<DataTypes> positionCoefficient = 19;
    static constexpr Real_t<DataTypes> restPositionCoefficient = 20;
    static constexpr Real_t<DataTypes> freePositionCoefficient = 5;
    static constexpr Real_t<DataTypes> velocityCoefficient = 12;
    static constexpr Real_t<DataTypes> forceCoefficient = 63;
    static constexpr Real_t<DataTypes> freeVelocityCoefficient = 78;
    
    void doSetUp() override
    {
        m_mechanicalObject = core::objectmodel::New<MO>();

        m_mechanicalObject->x0.forceSet();
        m_mechanicalObject->f.forceSet();
        m_mechanicalObject->xfree.forceSet();
        m_mechanicalObject->vfree.forceSet();
        
        m_mechanicalObject->resize(10);

        setVecValues<core::vec_id::write_access::position>(positionCoefficient);
        checkVecValues<core::vec_id::read_access::position>(positionCoefficient);

        setVecValues<core::vec_id::write_access::restPosition>(restPositionCoefficient);
        checkVecValues<core::vec_id::read_access::restPosition>(restPositionCoefficient);

        setVecValues<core::vec_id::write_access::freePosition>(freePositionCoefficient);
        checkVecValues<core::vec_id::read_access::freePosition>(freePositionCoefficient);

        setVecValues<core::vec_id::write_access::velocity>(velocityCoefficient);
        checkVecValues<core::vec_id::read_access::velocity>(velocityCoefficient);

        setVecValues<core::vec_id::write_access::force>(forceCoefficient);
        checkVecValues<core::vec_id::read_access::force>(forceCoefficient);
        
        setVecValues<core::vec_id::write_access::freeVelocity>(freeVelocityCoefficient);
        checkVecValues<core::vec_id::read_access::freeVelocity>(freeVelocityCoefficient);
    }

    template<core::TVecId vtype>
    void setVecValues(Real_t<DataTypes> coefficient = 1) const
    {
        unsigned int index {};
        auto vec = sofa::helper::getWriteOnlyAccessor(*m_mechanicalObject->write(vtype));
        ASSERT_EQ(vec.size(), 10);
        for (auto& v : vec)
        {
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                v[i] = static_cast<Real_t<DataTypes>>(index) * coefficient;
            }
            ++index;
        }
    }

    template<core::TVecId vtype>
    void checkVecValues(Real_t<DataTypes> coefficient = 1) const
    {
        unsigned int index {};
        auto vec = sofa::helper::getReadAccessor(*m_mechanicalObject->read(vtype));
        ASSERT_EQ(vec.size(), 10);
        for (auto& v : vec)
        {
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                EXPECT_FLOATINGPOINT_EQ(v[i], static_cast<Real_t<DataTypes>>(index) * coefficient);
            }
            ++index;
        }
    }

    template<core::TVecId vtype>
    void checkVecSpatialValues(Real_t<DataTypes> coefficient = 1) const
    {
        unsigned int index {};
        auto vec = sofa::helper::getReadAccessor(*m_mechanicalObject->read(vtype));
        ASSERT_EQ(vec.size(), 10);
        for (auto& v : vec)
        {
            for (std::size_t i = 0; i < DataTypes::spatial_dimensions; ++i)
            {
                EXPECT_FLOATINGPOINT_EQ(v[i], static_cast<Real_t<DataTypes>>(index) * coefficient);
            }
            ++index;
        }
    }

    bool v_null() const
    {
        //check that an error is emitted
        {
            EXPECT_MSG_EMIT(Error);
            m_mechanicalObject->vOp(nullptr, core::VecId::null());
        }

        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
        
        return true;
    }

    void resetPosition() const
    {
        //check that an error is not emitted
        {
            EXPECT_MSG_NOEMIT(Error);
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position);
        }

        //position is reset
        for (auto& position : m_mechanicalObject->readPositions())
        {
            static Coord_t<DataTypes> defaulConstructedCoord;
            for (std::size_t i = 0; i < position.size(); ++i)
            {
                EXPECT_FLOATINGPOINT_EQ(position[i], defaulConstructedCoord[i]);
            }
        }

        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void resetVelocity() const
    {
        //check that an error is not emitted
        {
            EXPECT_MSG_NOEMIT(Error);
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity);
        }

        //velocity is reset
        for (auto& velocity : m_mechanicalObject->readVelocities())
        {
            static Deriv_t<DataTypes> defaulConstructedDeriv;
            for (std::size_t i = 0; i < velocity.size(); ++i)
            {
                EXPECT_FLOATINGPOINT_EQ(velocity[i], defaulConstructedDeriv[i]);
            }
        }

        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
    }

    void multiplyByScalarPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v *= f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::ConstVecId::null(), core::vec_id::read_access::position, 2._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient * 2_sreal);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void multiplyByScalarVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v *= f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::ConstVecId::null(), core::vec_id::read_access::velocity, 2._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient * 2_sreal);
    }

    void equalOtherMultiplyByScalarPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::ConstVecId::null(), core::vec_id::read_access::restPosition, 2._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient * 2);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalOtherMultiplyByScalarVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::ConstVecId::null(), core::vec_id::read_access::force, 2._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(forceCoefficient * 2);
    }

    void equalOtherMultiplyByScalarPositionMix() const
    {
        //v = b*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::ConstVecId::null(), core::vec_id::read_access::velocity, 2._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(isRigid ? positionCoefficient : velocityCoefficient * 2_sreal);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalOtherMultiplyByScalarVelocityMix() const
    {
        //v = b*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::ConstVecId::null(), core::vec_id::read_access::position, 2._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : positionCoefficient * 2_sreal);
    }

    void equalOtherPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::ConstVecId::null(), 1._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalOtherPositionMix() const
    {
        //v = a
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::velocity, core::ConstVecId::null(), 1._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(isRigid ? positionCoefficient : velocityCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalOtherVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::ConstVecId::null(), 1._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(forceCoefficient);
    }

    void equalOtherVelocityMix() const
    {
        //v = a
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::restPosition, core::ConstVecId::null(), 1._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : restPositionCoefficient);
    }

    void plusEqualOtherPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += b
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::position, core::vec_id::read_access::restPosition, 1._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient + restPositionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void plusEqualOtherPositionMix() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += b
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::position, core::vec_id::read_access::freeVelocity, 1._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient + freeVelocityCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }
    
    void plusEqualOtherVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += b
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::velocity, core::vec_id::read_access::force, 1._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient + forceCoefficient);
    }

    void plusEqualOtherVelocityMix() const
    {
        //v += b
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::velocity, core::vec_id::read_access::position, 1._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : velocityCoefficient + positionCoefficient);
    }
    
    void plusEqualOtherMultipliedByScalarPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::position, core::vec_id::read_access::restPosition, 2._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient + restPositionCoefficient * 2);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void plusEqualOtherMultipliedByScalarPositionMix() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::position, core::vec_id::read_access::freeVelocity, 2._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient + freeVelocityCoefficient * 2);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }
    
    void plusEqualOtherMultipliedByScalarVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::velocity, core::vec_id::read_access::force, 2._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient + forceCoefficient * 2);
    }

    void plusEqualOtherMultipliedByScalarVelocityMix() const
    {
        //v += b*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::velocity, core::vec_id::read_access::freePosition, 2._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : velocityCoefficient + freePositionCoefficient * 2);
    }

    void plusEqualOtherPosition_2() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += a
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::vec_id::read_access::position, 1._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient + restPositionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void plusEqualOtherPositionMix_2() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += a
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::freeVelocity, core::vec_id::read_access::position, 1._sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient + freeVelocityCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }
    
    void plusEqualOtherVelocity_2() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v += a
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::vec_id::read_access::velocity, 1._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient + forceCoefficient);
    }

    void plusEqualOtherVelocityMix_2() const
    {
        //v += a
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::freePosition, core::vec_id::read_access::velocity, 1._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : velocityCoefficient + freePositionCoefficient);
    }

    void multipliedByScalarThenAddOtherPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+v*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::vec_id::read_access::position, 3_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient + positionCoefficient * 3);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void multipliedByScalarThenAddOtherPositionMix() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+v*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::freeVelocity, core::vec_id::read_access::position, 3_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(freeVelocityCoefficient + positionCoefficient * 3);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void multipliedByScalarThenAddOtherVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+v*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::vec_id::read_access::velocity, 7._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient * 7 + forceCoefficient);
    }

    void multipliedByScalarThenAddOtherVelocityMix() const
    {
        //v = a+v*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::freePosition, core::vec_id::read_access::velocity, 7._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : freePositionCoefficient + velocityCoefficient * 7);
    }

    void equalSumPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+b
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::vec_id::read_access::freePosition, 1_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient + freePositionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalSumPositionMix1() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+b
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::vec_id::read_access::freeVelocity, 1_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient+freeVelocityCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalSumPositionMix2() const
    {
        //v = a+b
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::freeVelocity, core::vec_id::read_access::freePosition, 1_sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(isRigid ? positionCoefficient : freeVelocityCoefficient + freePositionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalSumVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+b
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::vec_id::read_access::freeVelocity, 1._sreal);
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(forceCoefficient + freeVelocityCoefficient);
    }

    void equalSumVelocityMix1() const
    {
        //v = a+b
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::position, core::vec_id::read_access::freeVelocity, 1._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : positionCoefficient + freeVelocityCoefficient);
    }

    void equalSumVelocityMix2() const
    {
        //v = a+b
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::vec_id::read_access::position, 1._sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : forceCoefficient + positionCoefficient);
    }

    void equalSumWithScalarPosition() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::vec_id::read_access::freePosition, 12_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient + freePositionCoefficient * 12);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalSumWithScalarPositionMix1() const
    {
        //v = a+b*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::velocity, core::vec_id::read_access::freePosition, 12_sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(isRigid ? positionCoefficient : velocityCoefficient + freePositionCoefficient * 12);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalSumWithScalarPositionMix2() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::position, core::vec_id::read_access::restPosition, core::vec_id::read_access::freeVelocity, 12_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(restPositionCoefficient + freeVelocityCoefficient * 12);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(velocityCoefficient);
    }

    void equalSumWithScalarVelocity() const
    {
        {
            EXPECT_MSG_NOEMIT(Error);
            //v = a+b*f
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::vec_id::read_access::freeVelocity, 12_sreal);
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(forceCoefficient + freeVelocityCoefficient * 12);
    }

    void equalSumWithScalarVelocityMix1() const
    {
        //v = a+b*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::position, core::vec_id::read_access::freeVelocity, 12_sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : positionCoefficient + freeVelocityCoefficient * 12);
    }

    void equalSumWithScalarVelocityMix2() const
    {
        //v = a+b*f
        const auto vop = [this]()
        {
            m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::force, core::vec_id::read_access::position, 12_sreal);
        };
        if constexpr (isRigid)
        {
            EXPECT_MSG_EMIT(Error);
            vop();
        }
        else
        {
            EXPECT_MSG_NOEMIT(Error);
            vop();
        }
        checkVecSpatialValues<sofa::core::vec_id::read_access::position>(positionCoefficient);
        checkVecValues<sofa::core::vec_id::read_access::velocity>(isRigid ? velocityCoefficient : forceCoefficient + positionCoefficient * 12);
    }

    void equalCoordDifference() const
    {
        // v = a-b
        m_mechanicalObject->vOp(nullptr, core::vec_id::write_access::velocity, core::vec_id::read_access::restPosition, core::vec_id::read_access::position, -1_sreal);

        unsigned int index {};
        auto vv = sofa::helper::getReadAccessor(*m_mechanicalObject->read(core::vec_id::read_access::velocity));
        auto va = sofa::helper::getReadAccessor(*m_mechanicalObject->read(core::vec_id::read_access::restPosition));
        auto vb = sofa::helper::getReadAccessor(*m_mechanicalObject->read(core::vec_id::read_access::position));

        ASSERT_EQ(vv.size(), 10);
        for (std::size_t i = 0; i < vv.size(); ++i)
        {
            const auto& v = vv[i];
            const auto diff = DataTypes::coordDifference(va[i], vb[i]);
            for (std::size_t j = 0; j < v.size(); ++j)
            {
                EXPECT_FLOATINGPOINT_EQ(v[j], diff[j])
            }
            ++index;
        }

        checkVecValues<core::vec_id::read_access::position>(positionCoefficient);
    }

    typename MO::SPtr m_mechanicalObject;
};

typedef ::testing::Types<
    defaulttype::Vec1Types,
    defaulttype::Vec2Types,
    defaulttype::Vec3Types,
    defaulttype::Rigid2Types,
    defaulttype::Rigid3Types
> DataTypesList;

TYPED_TEST_SUITE(MechanicalObjectVOpTest, DataTypesList);

TYPED_TEST(MechanicalObjectVOpTest, v_null)
{
    EXPECT_TRUE(this->v_null());
}

TYPED_TEST(MechanicalObjectVOpTest, resetPosition)
{
    this->resetPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, resetVelocity)
{
    this->resetVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, multiplyByScalarPosition)
{
    this->multiplyByScalarPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, multiplyByScalarVelocity)
{
    this->multiplyByScalarVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherMultiplyByScalarPosition)
{
    this->equalOtherMultiplyByScalarPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherMultiplyByScalarVelocity)
{
    this->equalOtherMultiplyByScalarVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherMultiplyByScalarPositionMix)
{
    this->equalOtherMultiplyByScalarPositionMix();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherMultiplyByScalarVelocityMix)
{
    this->equalOtherMultiplyByScalarVelocityMix();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherPosition)
{
    this->equalOtherPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherPositionMix)
{
    this->equalOtherPositionMix();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherVelocity)
{
    this->equalOtherVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, equalOtherVelocityMix)
{
    this->equalOtherVelocityMix();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherPosition)
{
    this->plusEqualOtherPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherPositionMix)
{
    this->plusEqualOtherPositionMix();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherVelocity)
{
    this->plusEqualOtherVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherVelocityMix)
{
    this->plusEqualOtherVelocityMix();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherMultipliedByScalarPosition)
{
    this->plusEqualOtherMultipliedByScalarPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherMultipliedByScalarPositionMix)
{
    this->plusEqualOtherMultipliedByScalarPositionMix();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherMultipliedByScalarVelocity)
{
    this->plusEqualOtherMultipliedByScalarVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherMultipliedByScalarVelocityMix)
{
    this->plusEqualOtherMultipliedByScalarVelocityMix();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherPosition_2)
{
    this->plusEqualOtherPosition_2();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherPositionMix_2)
{
    this->plusEqualOtherPositionMix_2();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherVelocity_2)
{
    this->plusEqualOtherVelocity_2();
}

TYPED_TEST(MechanicalObjectVOpTest, plusEqualOtherVelocityMix_2)
{
    this->plusEqualOtherVelocityMix_2();
}

TYPED_TEST(MechanicalObjectVOpTest, multipliedByScalarThenAddOtherPosition)
{
    this->multipliedByScalarThenAddOtherPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, multipliedByScalarThenAddOtherPositionMix)
{
    this->multipliedByScalarThenAddOtherPositionMix();
}

TYPED_TEST(MechanicalObjectVOpTest, multipliedByScalarThenAddOtherVelocity)
{
    this->multipliedByScalarThenAddOtherVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, multipliedByScalarThenAddOtherVelocityMix)
{
    this->multipliedByScalarThenAddOtherVelocityMix();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumPosition)
{
    this->equalSumPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumPositionMix1)
{
    this->equalSumPositionMix1();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumPositionMix2)
{
    this->equalSumPositionMix2();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumVelocity)
{
    this->equalSumVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumVelocityMix1)
{
    this->equalSumVelocityMix1();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumVelocityMix2)
{
    this->equalSumVelocityMix2();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumWithScalarPosition)
{
    this->equalSumWithScalarPosition();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumWithScalarPositionMix1)
{
    this->equalSumWithScalarPositionMix1();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumWithScalarPositionMix2)
{
    this->equalSumWithScalarPositionMix2();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumWithScalarVelocity)
{
    this->equalSumWithScalarVelocity();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumWithScalarVelocityMix1)
{
    this->equalSumWithScalarVelocityMix1();
}

TYPED_TEST(MechanicalObjectVOpTest, equalSumWithScalarVelocityMix2)
{
    this->equalSumWithScalarVelocityMix2();
}

TYPED_TEST(MechanicalObjectVOpTest, equalCoordDifference)
{
    this->equalCoordDifference();
}

}