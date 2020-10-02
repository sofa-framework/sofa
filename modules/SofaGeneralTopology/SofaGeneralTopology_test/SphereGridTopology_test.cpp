#include <SofaTest/Sofa_test.h>
#include <SofaTest/TestMessageHandler.h>
#include <SofaGeneralTopology/SphereGridTopology.h>

namespace sofa
{

using namespace sofa::component::topology;

struct SphereGridTopology_test : public Sofa_test<>
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

        SphereGridTopology::SPtr sphereGrid = sofa::core::objectmodel::New<SphereGridTopology>(5, 5, 5);
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
    SphereGridTopology::SPtr sphereGrid = sofa::core::objectmodel::New<SphereGridTopology>(nx, ny, nz);
    sphereGrid->init();

    EXPECT_EQ(sphereGrid->getNbPoints(), nx*ny*nz);

    int nbHexa = (nx-1)*(ny-1)*(nz-1);
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
    SphereGridTopology::SPtr sphereGrid = sofa::core::objectmodel::New<SphereGridTopology>(nx, ny, nz);
    sphereGrid->init();

    int sphereSize = nx*ny*nz;
    int halfsphereSize = std::floor(sphereSize*0.5);

    for (int i=0; i<halfsphereSize; ++i){
        sofa::defaulttype::Vector3 p0 = sphereGrid->getPoint(i);
        sofa::defaulttype::Vector3 p1 = sphereGrid->getPoint(sphereSize-1-i);

        for (int j=0; j<3; ++j)
        {
            if (p0[j]+p1[j] > 0.001) // real numerical error
                EXPECT_EQ(p0[j]+p1[j], 0.0);
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
