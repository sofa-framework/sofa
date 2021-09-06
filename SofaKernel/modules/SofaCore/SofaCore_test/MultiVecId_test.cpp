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


#include <sofa/core/MultiVecId[V_ALL].h>
#include <sofa/core/MultiVecId[V_COORD].h>
#include <sofa/core/MultiVecId[V_DERIV].h>
#include <sofa/core/MultiVecId[V_MATDERIV].h>
using sofa::core::TMultiVecId;
using sofa::core::V_DERIV;
using sofa::core::V_COORD;
using sofa::core::V_MATDERIV;
using sofa::core::V_ALL;
using sofa::core::V_READ;
using sofa::core::V_WRITE;

#include <sofa/core/MultiVecId[V_ALL].h>

class MultiVecId_test: public BaseTest{};

template<class T>
void testConstructionBehavior()
{
    T id;
    ASSERT_TRUE(id.isNull()) << "After creation the MultiVecId is null";
    ASSERT_FALSE(id.hasIdMap()) << "After creation the MultiVecId has no idMap";

    id.assign(T::MyVecId::null());
    ASSERT_TRUE(id.isNull());

    typename T::MyVecId defaultId {10};
    id.assign(defaultId);
    ASSERT_EQ(id.getDefaultId(), defaultId);
    ASSERT_FALSE(id.isNull()) << "After assign() the MultiVecId is not null has it has a default value";
    ASSERT_FALSE(id.hasIdMap()) << "After assign()) the MultiVecId has no idMap...because it was reseted";
}

TEST_F(MultiVecId_test, checkConstructionBehavior)
{
    testConstructionBehavior<TMultiVecId<V_COORD, V_WRITE>>();
    testConstructionBehavior<TMultiVecId<V_COORD, V_READ>>();
    testConstructionBehavior<TMultiVecId<V_DERIV, V_WRITE>>();
    testConstructionBehavior<TMultiVecId<V_DERIV, V_READ>>();
    testConstructionBehavior<TMultiVecId<V_MATDERIV, V_WRITE>>();
    testConstructionBehavior<TMultiVecId<V_MATDERIV, V_READ>>();
    testConstructionBehavior<TMultiVecId<V_ALL, V_WRITE>>();
    testConstructionBehavior<TMultiVecId<V_ALL, V_READ>>();
}

template<class T>
void testSetId()
{
    T id;
    typename T::MyVecId defaultId {10};
    typename T::MyVecId firstId {11};
    typename T::MyVecId secondId {12};
    sofa::core::BaseState* state {nullptr};

    // Check that after assignement the default id is now the one we have set;
    id.assign(defaultId);
    ASSERT_EQ(id.getDefaultId(), defaultId);

    // Check that after setting a state-id pair the map is fully created;
    id.setId(state,firstId);
    ASSERT_FALSE(id.isNull());
    ASSERT_TRUE(id.hasIdMap());
    ASSERT_EQ(id.getId(state), firstId);

    id.setId(state,secondId);
    ASSERT_FALSE(id.isNull());
    ASSERT_TRUE(id.hasIdMap());
    ASSERT_EQ(id.getId(state), secondId);
}

TEST_F(MultiVecId_test, checkSetId)
{
    testSetId<TMultiVecId<V_COORD, V_WRITE>>();
    testSetId<TMultiVecId<V_COORD, V_READ>>();
    testSetId<TMultiVecId<V_DERIV, V_WRITE>>();
    testSetId<TMultiVecId<V_DERIV, V_READ>>();
    testSetId<TMultiVecId<V_MATDERIV, V_WRITE>>();
    testSetId<TMultiVecId<V_MATDERIV, V_READ>>();
    testSetId<TMultiVecId<V_ALL, V_WRITE>>();
    testSetId<TMultiVecId<V_ALL, V_READ>>();
}

template<class T>
void testConstructionCopyBehavior()
{
    T idSrc;
    T idDst{idSrc};
    ASSERT_EQ(idSrc, idDst);
}

TEST_F(MultiVecId_test, checkConstructor)
{
    testConstructionCopyBehavior<TMultiVecId<V_COORD, V_WRITE>>();
    testConstructionCopyBehavior<TMultiVecId<V_COORD, V_READ>>();
    testConstructionCopyBehavior<TMultiVecId<V_DERIV, V_WRITE>>();
    testConstructionCopyBehavior<TMultiVecId<V_DERIV, V_READ>>();
    testConstructionCopyBehavior<TMultiVecId<V_MATDERIV, V_WRITE>>();
    testConstructionCopyBehavior<TMultiVecId<V_MATDERIV, V_READ>>();
    testConstructionCopyBehavior<TMultiVecId<V_ALL, V_WRITE>>();
    testConstructionCopyBehavior<TMultiVecId<V_ALL, V_READ>>();
}
