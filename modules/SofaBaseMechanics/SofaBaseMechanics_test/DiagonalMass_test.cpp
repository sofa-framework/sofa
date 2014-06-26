/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define MY_PARAM_TYPE(name, DType, MType) \
    struct name { \
        typedef DType DataType; \
        typedef MType MassType; \
    }; \

#include "Sofa_test.h"

#include <SofaBaseMechanics/DiagonalMass.h>

//TODO : Perform smart tests :) Infrastructure for multi templated tests is ok.

namespace sofa {

template <class T>
class DiagonalMass_test : public ::testing::Test
{
public :

    typedef typename T::DataType dt;
    typedef typename T::MassType mt;
    typedef sofa::component::mass::DiagonalMass<dt,mt> DiagonalMassType;
    typename DiagonalMassType::SPtr m;

    /**
     * Constructor call for each test
     */
    virtual void SetUp(){}
};

TYPED_TEST_CASE_P(DiagonalMass_test);

TYPED_TEST_P(DiagonalMass_test, fakeTest1)
{
    EXPECT_EQ(0,0);
}

TYPED_TEST_P(DiagonalMass_test, fakeTest2)
{
    EXPECT_EQ(1,1);
}

REGISTER_TYPED_TEST_CASE_P(DiagonalMass_test, fakeTest1, fakeTest2);

MY_PARAM_TYPE(Vec3dd, sofa::defaulttype::Vec3dTypes, double)
MY_PARAM_TYPE(Vec2dd, sofa::defaulttype::Vec2dTypes, double)
MY_PARAM_TYPE(Vec1dd, sofa::defaulttype::Vec1dTypes, double)

INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case1, DiagonalMass_test, Vec3dd);
INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case2, DiagonalMass_test, Vec2dd);
INSTANTIATE_TYPED_TEST_CASE_P(DiagonalMass_test_case3, DiagonalMass_test, Vec1dd);

} // namespace sofa
