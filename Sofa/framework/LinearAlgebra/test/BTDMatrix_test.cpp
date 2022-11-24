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
#include <sofa/linearalgebra/BTDMatrix.h>
#include <gtest/gtest.h>

namespace sofa
{

TEST(BTDMatrix, name)
{
    const std::string BTDMatrix6f{sofa::linearalgebra::BTDMatrix<6, float>::Name()};
    EXPECT_EQ(BTDMatrix6f, "BTDMatrix6f");

    const std::string BTDMatrix6d{sofa::linearalgebra::BTDMatrix<6, double>::Name()};
    EXPECT_EQ(BTDMatrix6d, "BTDMatrix6d");

    const std::string BTDMatrix3d{sofa::linearalgebra::BTDMatrix<3, double>::Name()};
    EXPECT_EQ(BTDMatrix3d, "BTDMatrix3d");
}


}
