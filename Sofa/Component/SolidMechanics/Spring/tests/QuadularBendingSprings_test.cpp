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
#include <sofa/component/solidmechanics/spring/QuadularBendingSprings.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::spring
{

using QuadularBS = sofa::component::solidmechanics::spring::QuadularBendingSprings<defaulttype::Vec3Types>;

class QuadularBendingSpringsTest : public QuadularBS
{
public:
    using EdgeInformation = QuadularBS::EdgeInformation;
};

TEST(QuadularBendingSpringsTest, EdgeInformationStreamOperators)
{

    QuadularBendingSpringsTest::EdgeInformation initialInfo;

    for (int i = 0; i < 2; ++i)
    {
        initialInfo.springs[i].edge = {i,i};
        initialInfo.springs[i].restLength = i ;
        initialInfo.springs[i].DfDx = QuadularBendingSpringsTest::Mat();
    }

    initialInfo.ks = 1;
    initialInfo.kd = 1;
    initialInfo.is_activated = true;
    initialInfo.is_initialized = true;

    std::stringstream buffer;
    buffer << initialInfo;

    QuadularBendingSpringsTest::EdgeInformation loadedInfo;
    buffer >> loadedInfo;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
            EXPECT_EQ(initialInfo.springs[i].edge[j], loadedInfo.springs[i].edge[j]);
        EXPECT_EQ(initialInfo.springs[i].restLength, loadedInfo.springs[i].restLength);
        EXPECT_EQ(initialInfo.springs[i].DfDx, loadedInfo.springs[i].DfDx);
    }
    EXPECT_EQ(initialInfo.ks, loadedInfo.ks);
    EXPECT_EQ(initialInfo.kd, loadedInfo.kd);
    EXPECT_EQ(initialInfo.is_activated, loadedInfo.is_activated);
    EXPECT_EQ(initialInfo.is_initialized, loadedInfo.is_initialized);

}

}
