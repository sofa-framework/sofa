#include <string>
#include <vector>
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest;

#include <SofaGeometry/Plane.h>
using sofageometry::Plane;

namespace
{

TEST(Plane, checkDefaultConstructor)
{
    Plane p ;
    ASSERT_EQ( p.distance, 0.0 );
    ASSERT_EQ( p.normal, sofageometry::Constants::XAxis);
}


}


