#include <string>
#include <vector>
#include <sofa/helper/testing/BaseTest.h>
using sofa::helper::testing::BaseTest;

#include <SofaGeometry/Ray.h>
using sofageometry::Ray;
using sofageometry::Vec3d ;

namespace
{

TEST(Ray, checkDefaultConstructor)
{
    Ray r ;
    ASSERT_EQ( r.origin, sofageometry::Constants::Origin );
    ASSERT_EQ( r.direction, sofageometry::Constants::XAxis);
}


}


