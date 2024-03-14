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

using sofa::testing::BaseTest;

#include <sofa/component/topology/container/grid/RegularGridTopology.h>

using sofa::core::objectmodel::New;
using sofa::type::Vec3;
using namespace sofa::component::topology::container::grid;
using namespace sofa::testing;

struct RegularGridTopology_test : public BaseTest,
                                  public ::testing::WithParamInterface<std::vector<int>>
{
    bool regularGridCreation();
    bool regularGridPosition();
    bool regularGridFindPoint();

    bool regularGridSize(const std::vector<int>& p, bool fromTriangleList);
};

bool RegularGridTopology_test::regularGridCreation()
{
    // Creating a good Grid in 3D
    const RegularGridTopology::SPtr regGrid3 = New<RegularGridTopology>(5, 5, 5);
    EXPECT_NE(regGrid3, nullptr);
    EXPECT_EQ(regGrid3->d_p0.getValue(), Vec3(0.0_sreal, 0.0_sreal, 0.0_sreal));
    EXPECT_EQ(regGrid3->d_cellWidth.getValue(), 0.0);

    // Creating a good Grid in 2D
    const RegularGridTopology::SPtr regGrid2 = New<RegularGridTopology>(5, 5, 1);
    EXPECT_NE(regGrid2, nullptr);
    EXPECT_EQ(regGrid2->d_p0.getValue(), Vec3(0.0_sreal, 0.0_sreal, 0.0_sreal));
    EXPECT_EQ(regGrid2->d_cellWidth.getValue(), 0.0);

    // Creating a good Grid in 3D
    const RegularGridTopology::SPtr regGrid1 = New<RegularGridTopology>(5, 1, 1);
    EXPECT_NE(regGrid1, nullptr);
    EXPECT_EQ(regGrid1->d_p0.getValue(), Vec3(0.0_sreal, 0.0_sreal, 0.0_sreal));
    EXPECT_EQ(regGrid1->d_cellWidth.getValue(), 0.0);

    return true;
}

bool RegularGridTopology_test::regularGridSize(const std::vector<int>& p, bool fromTriangleList)
{
    int nx = p[0];
    int ny = p[1];
    int nz = p[2];

    /// Creating a good Grid in 3D
    const RegularGridTopology::SPtr regGrid = New<RegularGridTopology>(nx, ny, nz);
    regGrid->d_computeTriangleList.setValue(fromTriangleList);
    regGrid->init();

    /// The input was not valid...the default data should be used.
    if (p[4] == 1)
    {
        nx = 2;
        ny = 2;
        nz = 2;
    }

    /// check topology
    int nbHexa = (nx - 1) * (ny - 1) * (nz - 1);
    int nbQuads = (nx - 1) * (ny - 1) * nz + (nx - 1) * ny * (nz - 1) + nx * (ny - 1) * (nz - 1);

    /// Dimmension invariant assumption
    EXPECT_EQ(regGrid->getNbPoints(), nx * ny * nz);
    if (fromTriangleList)
    {
        const int nbEgdes = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1) + nbQuads;
        EXPECT_EQ(regGrid->getNbEdges(), nbEgdes);
    }
    else
    {
        const int nbEgdes = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
        EXPECT_EQ(regGrid->getNbEdges(), nbEgdes);
    }

    /// Compute the dimmension.
    const int d = (p[0] == 1) + (p[1] == 1) + (p[2] == 1); /// Check if there is reduced dimmension
    const int e = (p[0] <= 0) + (p[1] <= 0) + (p[2] <= 0); /// Check if there is an error
    if (e == 0)
    {
        if (d == 0)
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_3D);
        }
        else if (d == 1)
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_2D);
            nbHexa = 0;
        }
        else if (d == 2)
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_1D);
            nbHexa = 0;
            nbQuads = 0;
        }
        else
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_nullptr);
        }
    }
    EXPECT_EQ(regGrid->getNbHexahedra(), nbHexa);
    EXPECT_EQ(regGrid->getNbQuads(), nbQuads);
    return true;
}

bool RegularGridTopology_test::regularGridPosition()
{
    int nx = 8;
    int ny = 8;
    int nz = 5;
    const RegularGridTopology::SPtr regGrid = New<RegularGridTopology>(nx, ny, nz);
    regGrid->init();

    // Check first circle with
    sofa::type::Vec3 p0 = regGrid->getPoint(0);
    sofa::type::Vec3 p1 = regGrid->getPoint(nx - 1);
    // Check first point
    EXPECT_LE(p0[0], 0.0001);
    EXPECT_EQ(p0[0], p0[1]);

    // check last point of first line
    EXPECT_EQ(p0[0], -p1[0]);
    EXPECT_EQ(p0[1], p1[1]);

    // check first level
    EXPECT_EQ(p0[2], 0);

    // check last point of first level
    sofa::type::Vec3 p1Last = regGrid->getPoint(nx * ny - 1);
    EXPECT_LE(p1Last[0], 0.0001);
    EXPECT_EQ(p1[0], p1Last[0]);
    EXPECT_EQ(p1[1], -p1Last[1]);

    // Check first point of last level of the regular
    sofa::type::Vec3 p0Last = regGrid->getPoint(nx * ny * (nz - 1));
    EXPECT_EQ(p0Last[0], p0[0]);
    EXPECT_EQ(p0Last[1], p0[1]);

    return true;
}

bool RegularGridTopology_test::regularGridFindPoint()
{
    using Dimension = sofa::type::Vec3i;
    using BoundingBox = sofa::type::BoundingBox;
    using Coordinates = sofa::type::Vec3;
    using Epsilon = float;

    // 3D grid with 3x3x3=27 cells, each of dimension 1x1x1,  starting at {1,1,1}
    // and ending at {4,4,4}
    const auto grid = New<RegularGridTopology>(
                    Dimension{ 4, 4, 4 },
                    BoundingBox( Coordinates{ 1., 1., 1. } /*min*/, Coordinates{ 4, 4, 4 } /*max*/ )
                );
    grid->init();

    EXPECT_EQ(grid->findPoint(Coordinates{ .4, .4, .4 }), sofa::InvalidID);
    // No margin set means anything position rounded to a valid
    // node will be returned
    EXPECT_EQ(grid->findPoint(Coordinates{ .51, .51, .51 }), 0);
    EXPECT_EQ(grid->findPoint(Coordinates{ .51, .51, .51 }, Epsilon(0.01)), sofa::InvalidID);
    // Margin set means anything within a radius of 0.01*cell_size
    // will be returned
    EXPECT_EQ(grid->findPoint(Coordinates{ 1., 1., 1. }), 0); // First node of the grid
    EXPECT_EQ(grid->findPoint(Coordinates{ 4., 4., 4. }), 63); // Last node of the grid
    EXPECT_EQ(grid->findPoint(Coordinates{ 4.49, 4.49, 4.49 }), 63);
    EXPECT_EQ(grid->findPoint(Coordinates{ 4.51, 4, 4 }), sofa::InvalidID);
    EXPECT_EQ(grid->findPoint(Coordinates{ 4., 4.51, 4 }),sofa::InvalidID);
    EXPECT_EQ(grid->findPoint(Coordinates{ 4., 4, 4.51 }),sofa::InvalidID);
    return true;
}

TEST_F(RegularGridTopology_test, regularGridCreation)
{
    ASSERT_TRUE(regularGridCreation());
}
TEST_F(RegularGridTopology_test, regularGridPosition)
{
    ASSERT_TRUE(regularGridPosition());
}
TEST_F(RegularGridTopology_test, regularGridFindPoint)
{
    ASSERT_TRUE(regularGridFindPoint());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Test on various dimmensions
///
////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<int>> dimvalues = {
    /// The first three values are for the dimmension of the grid.
    /// The fourth is to encode if we need to catch a Warning message
    /// The fith is to indicate that the component should be initialized with
    /// the default values of
    /// 2-2-2
    {5,5,5, 0, 0},
    {2,2,2, 0, 0},
    {5,5,1, 0, 0},
    {5,1,5, 0, 0},
    {1,5,5, 0, 0},
    {1,1,5, 0, 0},
    {5,1,5, 0, 0},
    {5,1,1, 0, 0},
    {1,1,1, 0, 0},

    {0,0,0, 1, 1},
    {1,0,0, 1, 1},
    {1,1,0, 1, 1},
    {1,1,1, 0, 0},
    {0,1,1, 1, 1},
    {0,0,1, 1, 1},

    {6,0,0, 1, 1},
    {6,1,0, 1, 1},
    {1,1,6, 0, 0},
    {0,6,1, 1, 1},
    {6,0,6, 1, 1},

    {-5,5,5, 1, 1},
    {5,5,-1, 1, 1},
    {5,-1,5, 1, 1},
    {1,5,-5, 1, 1},
    {1,1,5,  0, 0},
    {-5,1,5, 1, 1},
    {5,1,-1, 1, 1},
    {-2,1,1, 1, 1},
};

TEST_P(RegularGridTopology_test, regularGridSizeComputeEdgeFromTriangle)
{
    /// We check if this test should returns a warning.
    if (GetParam()[3] == 1)
    {
        EXPECT_MSG_EMIT(Warning);
        ASSERT_TRUE(regularGridSize(GetParam(), true));
    }
    else
    {
        ASSERT_TRUE(regularGridSize(GetParam(), true));
    }
}

TEST_P(RegularGridTopology_test, regularGridSize)
{
    /// We check if this test should returns a warning.
    if (GetParam()[3] == 1)
    {
        EXPECT_MSG_EMIT(Warning);
        ASSERT_TRUE(regularGridSize(GetParam(), false));
    }
    else
    {
        ASSERT_TRUE(regularGridSize(GetParam(), false));
    }
}

INSTANTIATE_TEST_SUITE_P(regularGridSize3D,
                        RegularGridTopology_test,
                        ::testing::ValuesIn(dimvalues));
