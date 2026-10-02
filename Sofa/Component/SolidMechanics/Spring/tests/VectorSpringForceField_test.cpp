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
#include <sofa/component/solidmechanics/spring/VectorSpringForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::spring
{

using VectorSpringFF = sofa::component::solidmechanics::spring::VectorSpringForceField<defaulttype::Vec3Types>;

class VectorSpringForceFieldTest : public VectorSpringFF
{
public:
    using Spring = VectorSpringFF::Spring;
};

TEST(VectorSpringForceFieldTest, SpringStreamOperators)
{

    VectorSpringForceFieldTest::Spring initialInfo;

    initialInfo.ks = 1;
    initialInfo.kd = 1;
    initialInfo.restVector = VectorSpringForceFieldTest::Deriv();

    std::stringstream buffer;
    buffer << initialInfo;

    VectorSpringForceFieldTest::Spring loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.ks, loadedInfo.ks);
    EXPECT_EQ(initialInfo.kd, loadedInfo.kd);
    EXPECT_EQ(initialInfo.restVector, loadedInfo.restVector);

}

}
