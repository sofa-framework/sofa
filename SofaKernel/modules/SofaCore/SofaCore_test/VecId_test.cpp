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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest ;

#include <sofa/core/VecId.h>
using sofa::core::TVecId;
using sofa::core::V_DERIV;
using sofa::core::V_COORD;
using sofa::core::V_MATDERIV;
using sofa::core::V_ALL;
using sofa::core::V_READ;
using sofa::core::V_WRITE;

class VecId_test: public BaseTest
{
};

template<class T>
void testCommonBehavior()
{
    T id;
    ASSERT_EQ(id.getIndex(), 0);
    ASSERT_EQ(id.getType(), T::Type);
}

TEST_F(VecId_test, checkCommonBehavior)
{
    testCommonBehavior<TVecId<V_COORD, V_WRITE>>();
    testCommonBehavior<TVecId<V_COORD, V_READ>>();
    testCommonBehavior<TVecId<V_DERIV, V_WRITE>>();
    testCommonBehavior<TVecId<V_DERIV, V_READ>>();
    testCommonBehavior<TVecId<V_MATDERIV, V_WRITE>>();
    testCommonBehavior<TVecId<V_MATDERIV, V_READ>>();
    testCommonBehavior<TVecId<V_ALL, V_WRITE>>();
    testCommonBehavior<TVecId<V_ALL, V_READ>>();
}

template<class T>
void testGetName(unsigned int index, std::string value)
{
    T id{index};
    ASSERT_EQ(id.getName(), value);
}

template<class T1, class T2>
void testGetNameVecAll(unsigned int index, std::string value)
{
    T2 id2{index};
    T1 id1{id2};
    ASSERT_EQ(id1.getName(), value);
}

TEST_F(VecId_test, checkGetName)
{
    testGetName<TVecId<V_COORD, V_WRITE>>(0,"null(V_COORD)");
    testGetName<TVecId<V_DERIV, V_WRITE>>(0,"null(V_DERIV)");
    testGetName<TVecId<V_MATDERIV, V_WRITE>>(0,"null(V_MATDERIV)");
    testGetName<TVecId<V_ALL, V_WRITE>>(0,"null(V_ALL)");

    testGetName<TVecId<V_COORD, V_WRITE>>(2,"restPosition(V_COORD)");
    testGetName<TVecId<V_DERIV, V_WRITE>>(2,"resetVelocity(V_DERIV)");
    testGetName<TVecId<V_MATDERIV, V_WRITE>>(2,"nonHolonomic(V_MATDERIV)");
    testGetName<TVecId<V_ALL, V_WRITE>>(2,"2(V_ALL)");

    testGetNameVecAll<TVecId<V_ALL, V_WRITE>, TVecId<V_COORD, V_WRITE>>(2,"restPosition(V_ALL->V_COORD)");
    testGetNameVecAll<TVecId<V_ALL, V_WRITE>, TVecId<V_DERIV, V_WRITE>>(2,"resetVelocity(V_ALL->V_DERIV)");
    testGetNameVecAll<TVecId<V_ALL, V_WRITE>, TVecId<V_MATDERIV, V_WRITE>>(2,"nonHolonomic(V_ALL->V_MATDERIV)");
    testGetNameVecAll<TVecId<V_ALL, V_WRITE>, TVecId<V_ALL, V_WRITE>>(2,"2(V_ALL)");
}

template<class T>
void testConstructionCopyBehavior()
{
    T idSrc;
    T idDst{idSrc};
    ASSERT_EQ(idSrc, idDst);
}

TEST_F(VecId_test, checkConstructor)
{
    testConstructionCopyBehavior<TVecId<V_COORD, V_WRITE>>();
    testConstructionCopyBehavior<TVecId<V_COORD, V_READ>>();
    testConstructionCopyBehavior<TVecId<V_DERIV, V_WRITE>>();
    testConstructionCopyBehavior<TVecId<V_DERIV, V_READ>>();
    testConstructionCopyBehavior<TVecId<V_MATDERIV, V_WRITE>>();
    testConstructionCopyBehavior<TVecId<V_MATDERIV, V_READ>>();
    testConstructionCopyBehavior<TVecId<V_ALL, V_WRITE>>();
    testConstructionCopyBehavior<TVecId<V_ALL, V_READ>>();
}

template<class T>
void testNullBehavior(){
    T id;
    T n = T::null();
    T nonNull {123};
    ASSERT_TRUE(n.isNull());
    ASSERT_TRUE(id.isNull());
    ASSERT_FALSE(nonNull.isNull());
}

TEST_F(VecId_test, checkNullBehavior)
{
    testNullBehavior<TVecId<V_COORD, V_WRITE>>();
    testNullBehavior<TVecId<V_COORD, V_READ>>();
    testNullBehavior<TVecId<V_DERIV, V_WRITE>>();
    testNullBehavior<TVecId<V_DERIV, V_READ>>();
    testNullBehavior<TVecId<V_MATDERIV, V_WRITE>>();
    testNullBehavior<TVecId<V_MATDERIV, V_READ>>();
    testNullBehavior<TVecId<V_ALL, V_WRITE>>();
    testNullBehavior<TVecId<V_ALL, V_READ>>();
}

template<class DestType, class SrcType>
void testAvailability()
{
    SrcType idSrc{SrcType::null()};
    DestType idDst{idSrc};
    ASSERT_EQ(idSrc, idDst);
}

TEST_F(VecId_test, checkAvailability)
{
    testAvailability<TVecId<V_COORD, V_WRITE>, TVecId<V_COORD, V_WRITE>>();
    testAvailability<TVecId<V_COORD, V_READ>, TVecId<V_COORD, V_READ>>();
    testAvailability<TVecId<V_DERIV, V_WRITE>, TVecId<V_DERIV, V_WRITE>>();
    testAvailability<TVecId<V_DERIV, V_READ>, TVecId<V_DERIV, V_READ>>();
    testAvailability<TVecId<V_MATDERIV, V_WRITE>, TVecId<V_MATDERIV, V_WRITE>>();
    testAvailability<TVecId<V_MATDERIV, V_READ>, TVecId<V_MATDERIV, V_READ>>();
    testAvailability<TVecId<V_ALL, V_WRITE>, TVecId<V_ALL, V_WRITE>>();
    testAvailability<TVecId<V_ALL, V_READ>, TVecId<V_ALL, V_READ>>();

    testAvailability<TVecId<V_ALL, V_READ>, TVecId<V_DERIV, V_READ>>();
    testAvailability<TVecId<V_ALL, V_READ>, TVecId<V_COORD, V_READ>>();
    testAvailability<TVecId<V_ALL, V_READ>, TVecId<V_MATDERIV, V_READ>>();
    testAvailability<TVecId<V_ALL, V_WRITE>, TVecId<V_DERIV, V_WRITE>>();
    testAvailability<TVecId<V_ALL, V_WRITE>, TVecId<V_COORD, V_WRITE>>();
    testAvailability<TVecId<V_ALL, V_WRITE>, TVecId<V_MATDERIV, V_WRITE>>();
}

template<class SrcType, class DstType>
void testConversionThroughVall(unsigned int i, sofa::core::VecType outType)
{
    SrcType idSrc{i};
    TVecId<V_ALL, SrcType::Access> inBetween {idSrc};
    DstType idDst{inBetween};

    ASSERT_EQ(idDst.getType(), outType);
    ASSERT_EQ(idDst.getIndex(), i);
}

TEST_F(VecId_test, checkValidConversionFromV_ALL)
{
    testConversionThroughVall<TVecId<V_DERIV, V_READ>, TVecId<V_DERIV, V_READ>>(2,V_DERIV);
    testConversionThroughVall<TVecId<V_COORD, V_READ>, TVecId<V_COORD, V_READ>>(2,V_COORD);
    testConversionThroughVall<TVecId<V_MATDERIV, V_READ>, TVecId<V_MATDERIV, V_READ>>(2,V_MATDERIV);
}

TEST_F(VecId_test, checkInvalidConversionFromV_ALL)
{
    testConversionThroughVall<TVecId<V_COORD, V_READ>, TVecId<V_DERIV, V_READ>>(2,V_DERIV);
    testConversionThroughVall<TVecId<V_COORD, V_READ>, TVecId<V_MATDERIV, V_READ>>(2,V_MATDERIV);
    testConversionThroughVall<TVecId<V_DERIV, V_READ>, TVecId<V_COORD, V_READ>>(2,V_COORD);
    testConversionThroughVall<TVecId<V_DERIV, V_READ>, TVecId<V_MATDERIV, V_READ>>(2,V_MATDERIV);
}

template<class SrcType, class DstType>
void testConversion(unsigned int i, sofa::core::VecType outType)
{
    SrcType idSrc{i};
    DstType idDst{idSrc};

    ASSERT_EQ(idDst.getType(), outType);
    ASSERT_EQ(idDst.getIndex(), i);
}
TEST_F(VecId_test, checkConversionTFromV_ALL)
{
    testConversion<TVecId<V_DERIV, V_READ>, TVecId<V_ALL, V_READ>>(2,V_DERIV);
    testConversion<TVecId<V_COORD, V_READ>, TVecId<V_ALL, V_READ>>(2,V_COORD);
    testConversion<TVecId<V_MATDERIV, V_READ>, TVecId<V_ALL, V_READ>>(2,V_MATDERIV);
}

TEST_F(VecId_test, checkConversionBetweenTypes)
{
}
