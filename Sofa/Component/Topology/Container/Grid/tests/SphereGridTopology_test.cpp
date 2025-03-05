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

#include <sofa/component/topology/container/grid/SphereGridTopology.h>

namespace sofa
{

using namespace sofa::component::topology::container::grid;

struct SphereGridTopology_test : public BaseTest
{
    bool SphereGridCreation();
    bool SphereGridSize();
    bool SphereGridPosition();
};


bool SphereGridTopology_test::SphereGridCreation()
{
    // Creating a good Grid
    {
        EXPECT_MSG_NOEMIT(Warning) ;
        EXPECT_MSG_NOEMIT(Error) ;

        const SphereGridTopology::SPtr sphereGrid = sofa::core::objectmodel::New<SphereGridTopology>(5, 5, 5);
        EXPECT_NE(sphereGrid, nullptr);
        EXPECT_EQ(sphereGrid->d_radius.getValue(), 1.0);
    }

    // Creating a bad Grid
    {
        EXPECT_MSG_NOEMIT(Error) ;
        EXPECT_MSG_EMIT(Warning) ;

        SphereGridTopology::SPtr sphereGrid2 = sofa::core::objectmodel::New<SphereGridTopology>(-1, 0, 1);
    }

    return true;
}

bool SphereGridTopology_test::SphereGridSize()
{
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    // Creating a good Grid
    int nx = 5;
    int ny = 5;
    int nz = 5;
    const SphereGridTopology::SPtr sphereGrid = sofa::core::objectmodel::New<SphereGridTopology>(nx, ny, nz);
    sphereGrid->init();

    EXPECT_EQ(sphereGrid->getNbPoints(), nx*ny*nz);

    const int nbHexa = (nx-1)*(ny-1)*(nz-1);
    EXPECT_EQ(sphereGrid->getNbHexahedra(), nbHexa);

    return true;
}

bool SphereGridTopology_test::SphereGridPosition()
{
    EXPECT_MSG_NOEMIT(Warning) ;
    EXPECT_MSG_NOEMIT(Error) ;

    int nx = 6;
    int ny = 6;
    int nz = 5;
    const SphereGridTopology::SPtr sphereGrid = sofa::core::objectmodel::New<SphereGridTopology>(nx, ny, nz);
    sphereGrid->init();

    const int sphereSize = nx*ny*nz;
    const int halfsphereSize = std::floor(sphereSize*0.5);

    for (int i=0; i<halfsphereSize; ++i){
        sofa::type::Vec3 p0 = sphereGrid->getPoint(i);
        sofa::type::Vec3 p1 = sphereGrid->getPoint(sphereSize-1-i);

        for (int j=0; j<3; ++j)
        {
            if (p0[j]+p1[j] > 0.001) { // real numerical error
                EXPECT_EQ(p0[j] + p1[j], 0.0);
            }
        }
    }

    return true;
}

TEST_F(SphereGridTopology_test, SphereGridCreation ) {
    ASSERT_TRUE( SphereGridCreation());
}

TEST_F(SphereGridTopology_test, SphereGridSize ) {
    ASSERT_TRUE( SphereGridSize());
}

TEST_F(SphereGridTopology_test, SphereGridPosition ) {
    ASSERT_TRUE( SphereGridPosition());
}

}
