#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>


#include <SofaGeneralTopology/CylinderGridTopology.h>

namespace sofa
{

using namespace sofa::component::topology;

struct CylinderGridTopology_test : public Sofa_test<>
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
        CylinderGridTopology::SPtr cylGrid = sofa::core::objectmodel::New<CylinderGridTopology>(5, 5, 5);
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
    CylinderGridTopology::SPtr cylGrid = sofa::core::objectmodel::New<CylinderGridTopology>(nx, ny, nz);
    cylGrid->init();

    EXPECT_EQ(cylGrid->getNbPoints(), nx*ny*nz);

    int nbHexa = (nx-1)*(ny-1)*(nz-1);
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
    CylinderGridTopology::SPtr cylGrid = sofa::core::objectmodel::New<CylinderGridTopology>(nx, ny, nz);
    cylGrid->init();

    SReal length = cylGrid->d_length.getValue();

    // Check first circle with
    sofa::defaulttype::Vector3 p0 = cylGrid->getPoint(0);
    sofa::defaulttype::Vector3 p1 = cylGrid->getPoint(nx-1);
    // Check first point
    EXPECT_NE(p0[0], 0);
    EXPECT_EQ(p0[0], p0[1]);

    // check last point of first line
    EXPECT_EQ(p0[0], -p1[0]);
    EXPECT_EQ(p0[1], p1[1]);

    // check first level
    EXPECT_EQ(p0[2], 0);

    // check last point of first level
    sofa::defaulttype::Vector3 p1Last = cylGrid->getPoint(nx*ny -1);
    EXPECT_NE(p1Last[0], 0);
    EXPECT_EQ(p1[0], p1Last[0]);
    EXPECT_EQ(p1[1], -p1Last[1]);

    // Check first point of last level of the cylinder
    sofa::defaulttype::Vector3 p0Last = cylGrid->getPoint(nx*ny*(nz-1));
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
