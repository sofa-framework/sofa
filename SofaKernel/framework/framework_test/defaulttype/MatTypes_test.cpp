/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <iostream>

#include <sofa/defaulttype/Mat.h>

#include <sofa/defaulttype/Quat.h>

/// Beuaark
#include "../../../applications/plugins/SofaTest/Sofa_test.h"

#include <gtest/gtest.h>

using namespace sofa;
using namespace sofa::helper;
using namespace sofa::defaulttype;

void test_transformInverse(Matrix4 const& M)
{
    Matrix4 M_inv;
    M_inv.transformInvert(M);
    Matrix4 res = M*M_inv;
    Matrix4 I;I.identity();
    EXPECT_MAT_NEAR(I, res, (SReal)1e-12);
}

TEST(MatTypesTest, transformInverse)
{
    test_transformInverse(Matrix4::s_identity);
    test_transformInverse(Matrix4::transformTranslation(Vector3(1.,2.,3.)));
    test_transformInverse(Matrix4::transformScale(Vector3(1.,2.,3.)));
    test_transformInverse(Matrix4::transformRotation(Quat::fromEuler(3.14/4.,3.14/2.,3.14/3.)));
}

TEST(MatTypesTest, setsub_vec)
{
    Matrix3 M = Matrix3::s_identity;
    Vector2 v(1.,2.);
    M.setsub(1,2,v);
    double exp[9]={1.,0.,0.,
                   0.,1.,1.,
                   0.,0.,2.};
    Matrix3 M_exp(exp);
    EXPECT_MAT_DOUBLE_EQ(M_exp, M);
}

TEST(MatTypesTest, isTransform)
{
    Matrix4 M;
    EXPECT_FALSE(M.isTransform());
    M.identity();
    EXPECT_TRUE(M.isTransform());
}
