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
#include <gtest/gtest.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/DataTypeInfo.h>

template<sofa::Size N, typename real>
void testName(const std::string& expectedName)
{
    using Deriv = sofa::defaulttype::RigidDeriv<N, real>;
    using Bloc = sofa::linearalgebra::matrix_bloc_traits<Deriv, sofa::Index >;
    EXPECT_EQ(std::string(Bloc::Name()), expectedName);
}

TEST(RigidDerivTest, Name)
{
    testName<3, float>("RigidDeriv3f");
    testName<3, double>("RigidDeriv3d");

    testName<2, float>("RigidDeriv2f");
    testName<2, double>("RigidDeriv2d");
}
