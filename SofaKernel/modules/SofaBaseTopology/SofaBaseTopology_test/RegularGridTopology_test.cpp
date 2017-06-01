#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test;

#include <SofaBaseTopology/RegularGridTopology.h>

using sofa::core::objectmodel::New ;
using sofa::defaulttype::Vector3 ;
using namespace sofa::component::topology;

struct RegularGridTopology_test :
        public Sofa_test<>,
        public ::testing::WithParamInterface<std::vector<int>>
{
    bool regularGridCreation();
    bool regularGridPosition();

    bool regularGridSize(const std::vector<int>& p);
};


bool RegularGridTopology_test::regularGridCreation()
{
    // Creating a good Grid in 3D
    RegularGridTopology::SPtr regGrid3 =New<RegularGridTopology>(5, 5, 5);
    EXPECT_NE(regGrid3, nullptr);
    EXPECT_EQ(regGrid3->d_p0.getValue(), Vector3(0.0f,0.0f,0.0f));
    EXPECT_EQ(regGrid3->d_cellWidth.getValue(), 0.0);

    // Creating a good Grid in 2D
    RegularGridTopology::SPtr regGrid2 =New<RegularGridTopology>(5, 5, 1);
    EXPECT_NE(regGrid2, nullptr);
    EXPECT_EQ(regGrid2->d_p0.getValue(), Vector3(0.0f,0.0f,0.0f));
    EXPECT_EQ(regGrid2->d_cellWidth.getValue(), 0.0);

    // Creating a good Grid in 3D
    RegularGridTopology::SPtr regGrid1 =New<RegularGridTopology>(5, 1, 1);
    EXPECT_NE(regGrid1, nullptr);
    EXPECT_EQ(regGrid1->d_p0.getValue(), Vector3(0.0f,0.0f,0.0f));
    EXPECT_EQ(regGrid1->d_cellWidth.getValue(), 0.0);

    return true;
}

bool RegularGridTopology_test::regularGridSize(const std::vector<int>& p)
{
    int nx=p[0];
    int ny=p[1];
    int nz=p[2];

    /// Creating a good Grid in 3D
    RegularGridTopology::SPtr regGrid =New<RegularGridTopology>(nx, ny, nz);
    regGrid->init();

    /// The input was not valid...the default data should be used.
    if(p[4]==1){
        nx = 2;
        ny = 2;
        nz = 2;
    }

    /// check topology
    int nbHexa = (nx-1)*(ny-1)*(nz-1);
    int nbQuads = (nx-1)*(ny-1)*nz+(nx-1)*ny*(nz-1)+nx*(ny-1)*(nz-1);
    int nbEgdes = (nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1);

    /// Dimmension invariant assumption
    EXPECT_EQ(regGrid->getNbPoints(), nx*ny*nz);
    EXPECT_EQ(regGrid->getNbEdges(), nbEgdes);

    /// Compute the dimmension.
    int d=(p[0]==1)+(p[1]==1)+(p[2]==1) ; /// Check if there is reduced dimmension
    int e=(p[0]<=0)+(p[1]<=0)+(p[2]<=0) ; /// Check if there is an error
    if(e==0){
        if(d==0)
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_3D) ;
        }
        else if(d==1)
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_2D) ;
            nbHexa = 0;
        }
        else if(d==2)
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_1D) ;
            nbHexa = 0;
            nbQuads = 0;
        }
        else
        {
            EXPECT_EQ(regGrid->getDimensions(), Grid_dimension::GRID_NULL) ;
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
    RegularGridTopology::SPtr regGrid =New<RegularGridTopology>(nx, ny, nz);
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
TEST_F(RegularGridTopology_test, regularGridPosition ) { ASSERT_TRUE( regularGridPosition()); }


////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Test on various dimmensions
///
////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<int>> dimvalues={
    /// The first three values are for the dimmension of the grid.
    /// The fourth is to encode if we need to catch a Warning message
    /// The fith is to indicate that the component should be initialized with the default values of
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

TEST_P(RegularGridTopology_test, regularGridSize )
{
    /// We check if this test should returns a warning.
    if(GetParam()[3]==1){
        {
            EXPECT_MSG_EMIT(Warning) ;
            ASSERT_TRUE( regularGridSize(GetParam()) );
        }
    }else{
        ASSERT_TRUE( regularGridSize(GetParam()) );
    }
}
INSTANTIATE_TEST_CASE_P(regularGridSize3D,
                        RegularGridTopology_test,
                        ::testing::ValuesIn(dimvalues));


