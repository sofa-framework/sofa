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

#include <sofa/component/topology/container/grid/CylinderGridTopology.h>

namespace sofa
{

using namespace sofa::component::topology::container::grid;

struct CylinderGridTopology_test : public BaseTest
{
    bool cylinderGridCreation();
    bool cylinderGridSize();
    bool cylinderGridPosition();
};


bool CylinderGridTopology_test::cylinderGridCreation()
{
    {
        EXPECT_MSG_NOEMIT(Error) ;
        EXPECT_MSG_NOEMIT(Warning) ;

        // Creating a good Grid
        const CylinderGridTopology::SPtr cylGrid = sofa::core::objectmodel::New<CylinderGridTopology>(5, 5, 5);
        EXPECT_NE(cylGrid, nullptr);
        EXPECT_EQ(cylGrid->d_radius.getValue(), 1.0);
        EXPECT_EQ(cylGrid->d_length.getValue(), 1.0);
    }


    {
        EXPECT_MSG_NOEMIT(Error) ;
        EXPECT_MSG_EMIT(Warning) ;

        // Creating a bad Grid
        CylinderGridTopology::SPtr cylGrid2 = sofa::core::objectmodel::New<CylinderGridTopology>(-1, 0, 1);
    }

    return true;
}

bool CylinderGridTopology_test::cylinderGridSize()
{
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    // Creating a good Grid
    int nx = 5;
    int ny = 5;
    int nz = 5;
    const CylinderGridTopology::SPtr cylGrid = sofa::core::objectmodel::New<CylinderGridTopology>(nx, ny, nz);
    cylGrid->init();

    EXPECT_EQ(cylGrid->getNbPoints(), nx*ny*nz);

    const int nbHexa = (nx-1)*(ny-1)*(nz-1);
    EXPECT_EQ(cylGrid->getNbHexahedra(), nbHexa);

    return true;
}

bool CylinderGridTopology_test::cylinderGridPosition()
{
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    int nx = 8;
    int ny = 8;
    int nz = 5;
    const CylinderGridTopology::SPtr cylGrid = sofa::core::objectmodel::New<CylinderGridTopology>(nx, ny, nz);
    cylGrid->init();

    const SReal length = cylGrid->d_length.getValue();

    // Check first circle with
    sofa::type::Vec3 p0 = cylGrid->getPoint(0);
    sofa::type::Vec3 p1 = cylGrid->getPoint(nx-1);
    // Check first point
    EXPECT_NE(p0[0], 0);
    EXPECT_EQ(p0[0], p0[1]);

    // check last point of first line
    EXPECT_EQ(p0[0], -p1[0]);
    EXPECT_EQ(p0[1], p1[1]);

    // check first level
    EXPECT_EQ(p0[2], 0);

    // check last point of first level
    sofa::type::Vec3 p1Last = cylGrid->getPoint(nx*ny -1);
    EXPECT_NE(p1Last[0], 0);
    EXPECT_EQ(p1[0], p1Last[0]);
    EXPECT_EQ(p1[1], -p1Last[1]);

    // Check first point of last level of the cylinder
    sofa::type::Vec3 p0Last = cylGrid->getPoint(nx*ny*(nz-1));
    EXPECT_EQ(p0Last[2], length);
    EXPECT_EQ(p0Last[0], p0[0]);
    EXPECT_EQ(p0Last[1], p0[1]);

    return true;
}

TEST_F(CylinderGridTopology_test, cylinderGridCreation ) {
    ASSERT_TRUE( cylinderGridCreation());
}

TEST_F(CylinderGridTopology_test, cylinderGridSize ) {
    ASSERT_TRUE( cylinderGridSize());
}

TEST_F(CylinderGridTopology_test, cylinderGridPosition ) {
    ASSERT_TRUE( cylinderGridPosition());
}

}
