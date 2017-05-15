#include <SofaTest/Sofa_test.h>
#include <SofaBaseTopology/RegularGridTopology.h>

namespace sofa
{

using namespace sofa::component::topology;

struct RegularGridTopology_test : public Sofa_test<>
{
    bool regularGridCreation();
    bool regularGridSize();
    bool regularGridPosition();
};


bool RegularGridTopology_test::regularGridCreation()
{
    // Creating a good Grid in 3D
    RegularGridTopology::SPtr regGrid3 = sofa::core::objectmodel::New<RegularGridTopology>(5, 5, 5);
    EXPECT_NE(regGrid3, nullptr);
    EXPECT_EQ(regGrid3->d_p0.getValue(), sofa::defaulttype::Vector3(0.0f,0.0f,0.0f));
    EXPECT_EQ(regGrid3->d_cellWidth.getValue(), 0.0);

    // Creating a good Grid in 2D
    RegularGridTopology::SPtr regGrid2 = sofa::core::objectmodel::New<RegularGridTopology>(5, 5, 1);
    EXPECT_NE(regGrid2, nullptr);
    EXPECT_EQ(regGrid2->d_p0.getValue(), sofa::defaulttype::Vector3(0.0f,0.0f,0.0f));
    EXPECT_EQ(regGrid2->d_cellWidth.getValue(), 0.0);

    // Creating a good Grid in 3D
    RegularGridTopology::SPtr regGrid1 = sofa::core::objectmodel::New<RegularGridTopology>(5, 1, 1);
    EXPECT_NE(regGrid1, nullptr);
    EXPECT_EQ(regGrid1->d_p0.getValue(), sofa::defaulttype::Vector3(0.0f,0.0f,0.0f));
    EXPECT_EQ(regGrid1->d_cellWidth.getValue(), 0.0);

    // Creating a bad Grid
    RegularGridTopology::SPtr regGrid0 = sofa::core::objectmodel::New<RegularGridTopology>(-1, 0, 1);
    // EXPECT_EQ(regGrid2, nullptr);

    return true;
}

bool RegularGridTopology_test::regularGridSize()
{
    int nx = 5;
    int ny = 5;
    int nz = 5;

    // Creating a good Grid in 3D
    RegularGridTopology::SPtr regGrid3 = sofa::core::objectmodel::New<RegularGridTopology>(nx, ny, nz);
    regGrid3->init();
    //regGrid3->getGridUpdate()->update();

    // check topology
    int nbHexa = (nx-1)*(ny-1)*(nz-1);
    int nbQuads = (nx-1)*(ny-1)*nz+(nx-1)*ny*(nz-1)+nx*(ny-1)*(nz-1);
    int nbEgdes = (nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1);
    EXPECT_EQ(regGrid3->getNbPoints(), nx*ny*nz);
    EXPECT_EQ(regGrid3->getNbHexahedra(), nbHexa);
    EXPECT_EQ(regGrid3->getNbQuads(), nbQuads);
    EXPECT_EQ(regGrid3->getNbEdges(), nbEgdes);


    // Creating a good Grid in 2D
    nz = 1;
    RegularGridTopology::SPtr regGrid2 = sofa::core::objectmodel::New<RegularGridTopology>(nx, ny, nz);
    regGrid2->init();

    // check topology
    nbQuads = (nx-1)*(ny-1)*nz+(nx-1)*ny*(nz-1)+nx*(ny-1)*(nz-1);
    nbEgdes = (nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1);
    EXPECT_EQ(regGrid2->getNbPoints(), nx*ny*nz);
    EXPECT_EQ(regGrid2->getNbHexahedra(), 0);
    EXPECT_EQ(regGrid2->getNbQuads(), nbQuads);
    EXPECT_EQ(regGrid2->getNbEdges(), nbEgdes);


    // Creating a good Grid in 2D
    nz = 1;
    ny = 1;
    RegularGridTopology::SPtr regGrid1 = sofa::core::objectmodel::New<RegularGridTopology>(nx, ny, nz);
    regGrid1->init();

    // check topology
    nbEgdes = (nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1);
    EXPECT_EQ(regGrid1->getNbPoints(), nx*ny*nz);
    EXPECT_EQ(regGrid1->getNbHexahedra(), 0);
    EXPECT_EQ(regGrid1->getNbQuads(), 0);
    EXPECT_EQ(regGrid1->getNbEdges(), nbEgdes);


    return true;
}

bool RegularGridTopology_test::regularGridPosition()
{
    int nx = 8;
    int ny = 8;
    int nz = 5;
    RegularGridTopology::SPtr regGrid = sofa::core::objectmodel::New<RegularGridTopology>(nx, ny, nz);
    regGrid->init();

    // Check first circle with
    sofa::defaulttype::Vector3 p0 = regGrid->getPoint(0);
    sofa::defaulttype::Vector3 p1 = regGrid->getPoint(nx-1);
    // Check first point
    EXPECT_LE(p0[0], 0.0001);
    EXPECT_EQ(p0[0], p0[1]);

    // check last point of first line
    EXPECT_EQ(p0[0], -p1[0]);
    EXPECT_EQ(p0[1], p1[1]);

    // check first level
    EXPECT_EQ(p0[2], 0);

    // check last point of first level
    sofa::defaulttype::Vector3 p1Last = regGrid->getPoint(nx*ny -1);
    EXPECT_LE(p1Last[0], 0.0001);
    EXPECT_EQ(p1[0], p1Last[0]);
    EXPECT_EQ(p1[1], -p1Last[1]);

    // Check first point of last level of the regular
    sofa::defaulttype::Vector3 p0Last = regGrid->getPoint(nx*ny*(nz-1));
    EXPECT_EQ(p0Last[0], p0[0]);
    EXPECT_EQ(p0Last[1], p0[1]);

    return true;
}

TEST_F(RegularGridTopology_test, regularGridCreation ) { ASSERT_TRUE( regularGridCreation()); }
TEST_F(RegularGridTopology_test, regularGridSize ) { ASSERT_TRUE( regularGridSize()); }
TEST_F(RegularGridTopology_test, regularGridPosition ) { ASSERT_TRUE( regularGridPosition()); }
}
