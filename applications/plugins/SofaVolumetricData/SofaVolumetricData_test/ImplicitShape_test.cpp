#include <SofaTest/Sofa_test.h>
typedef sofa::defaulttype::Vector3 Coord;

#include <SofaVolumetricData/components/implicit/ScalarField.h>
using sofa::component::implicit::ScalarField ;

#include <SofaVolumetricData/components/implicit/SphericalField.h>
using sofa::component::implicit::SphericalField ;

#include <SofaVolumetricData/components/implicit/DiscreteGridField.h>
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::component::implicit::DiscreteGridField ;

namespace
{

struct ImplicitShape_test : public sofa::Sofa_test<>
{
    bool SphericalFieldTest();
    bool DiscreteGridFieldTest();
};


bool ImplicitShape_test::SphericalFieldTest()
{
    SphericalField sphere_test;
    Coord p(1,1,2);
    EXPECT_EQ(sphere_test.eval(p),19);
    return true;
}


bool ImplicitShape_test::DiscreteGridFieldTest()
{
    DiscreteGridField dgc_test;
    dgc_test.setFilename("/path/shape.obj");
    Coord pmin(0,0,0), pmax(243,243,243);
    dgc_test.loadGrid(0,0,243,243,243,pmin,pmax);
    EXPECT_NE(dgc_test.grid,nullptr);
    return true;
}

TEST_F(ImplicitShape_test, SphericalFieldTest) { ASSERT_TRUE( SphericalFieldTest() ); }
TEST_F(ImplicitShape_test, DiscreteGridFieldTest) { ASSERT_TRUE( DiscreteGridFieldTest() ); }

}
