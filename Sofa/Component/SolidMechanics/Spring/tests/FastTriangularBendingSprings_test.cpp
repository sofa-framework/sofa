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
#include <sofa/component/solidmechanics/spring/FastTriangularBendingSprings.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::spring
{

using FastTriangularBS = sofa::component::solidmechanics::spring::FastTriangularBendingSprings<defaulttype::Vec3Types>;

class FastTriangularBendingSpringsTest : public FastTriangularBS
{
public:
    using EdgeSpring = FastTriangularBS::EdgeSpring;
};

TEST(FastTriangularBendingSpringsTest, EdgePressureStreamOperators)
{

    FastTriangularBendingSpringsTest::EdgeSpring initialInfo;

    for (int i = 0; i < 4; ++i)
        initialInfo.vid[i] = i;

    for (int i = 0; i < 4; ++i)
        initialInfo.alpha[i] = i;
    initialInfo.lambda = 1;
    initialInfo.is_activated = true;
    initialInfo.is_initialized = true;

    std::stringstream buffer;
    buffer << initialInfo;

    FastTriangularBendingSpringsTest::EdgeSpring loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.vid, loadedInfo.vid);
    EXPECT_EQ(initialInfo.alpha, loadedInfo.alpha);
    EXPECT_EQ(initialInfo.lambda, loadedInfo.lambda);
    EXPECT_EQ(initialInfo.is_activated, loadedInfo.is_activated);
    EXPECT_EQ(initialInfo.is_initialized, loadedInfo.is_initialized);

}

}
