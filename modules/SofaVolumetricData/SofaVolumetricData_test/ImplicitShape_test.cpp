//This file contain tests related to the implicit shapes

#include <SofaTest/Sofa_test.h>
#include <SofaVolumetricData/ImplicitShape.h>
#include <SofaVolumetricData/ImplicitSphere.h>
#include <SofaVolumetricData/DistanceGridComponent.h>
#include <sofa/core/objectmodel/BaseObject.h>

typedef sofa::defaulttype::Vector3 Coord;

struct ImplicitShape_test : public sofa::Sofa_test<>
{
    bool ImplicitSphereTest();
    bool DistanceGridComponentTest();
};


bool ImplicitShape_test::ImplicitSphereTest() {
    sofa::core::ImplicitSphere sphere_test;
    Coord p(1,1,2);
    EXPECT_EQ(sphere_test.eval(p),19);
    return true;
}


bool ImplicitShape_test::DistanceGridComponentTest() {
    sofa::core::DistanceGridComponent dgc_test;
    //need to put any file that DistanceGrid can handle
    dgc_test.setFilename("/home/tgosse/Bureau/monkey.obj");
    Coord pmin(0,0,0), pmax(243,243,243);
    dgc_test.loadGrid(0,0,243,243,243,pmin,pmax);
    EXPECT_NE(dgc_test.grid,nullptr);
    return true;
}

TEST_F(ImplicitShape_test, ImplicitSphereTest) { ASSERT_TRUE( ImplicitSphereTest()); }
TEST_F(ImplicitShape_test, DistanceGridComponentTest) { ASSERT_TRUE( DistanceGridComponentTest()); }
