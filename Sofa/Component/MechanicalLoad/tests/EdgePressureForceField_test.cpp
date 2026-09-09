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
#include <sofa/component/mechanicalload/EdgePressureForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::mechanicalload
{

using EdgePressureFF = sofa::component::mechanicalload::EdgePressureForceField<defaulttype::Vec3Types>;

class EdgePressureForceFieldTest : public EdgePressureFF
{
public:
    using EdgePressureInformation = EdgePressureFF::EdgePressureInformation;
};

TEST(EdgePressureForceFieldTest, EdgePressureInformationStreamOperators)
{

    EdgePressureForceFieldTest::EdgePressureInformation initialInfo;

    initialInfo.length = 1.0;

    initialInfo.force[0] = 1.0;
    initialInfo.force[1] = 2.0;
    initialInfo.force[2] = 3.0;

    std::stringstream buffer;
    buffer << initialInfo;

    EdgePressureForceFieldTest::EdgePressureInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.length, loadedInfo.length);

    EXPECT_EQ(initialInfo.force[0], loadedInfo.force[0]);
    EXPECT_EQ(initialInfo.force[1], loadedInfo.force[1]);
    EXPECT_EQ(initialInfo.force[2], loadedInfo.force[2]);

}

}
