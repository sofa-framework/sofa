#include <SofaTest/Sofa_test.h>
using sofa::defaulttype::Vec3d ;

#include <SofaImplicitField/components/geometry/SphericalField.h>
using sofa::component::geometry::SphericalField ;

namespace
{

class SphericalFieldTest : public sofa::Sofa_test<>
{
public:
    bool checkSphericalField();
    bool checkDiscreteGridField();
};


bool SphericalFieldTest::checkSphericalField()
{
    SphericalField sphere_test;
    Vec3d p(1,1,2);
    sphere_test.getValue(p) ;
    return true;
}


TEST_F(SphericalFieldTest, checkSphericalField) { ASSERT_TRUE( checkSphericalField() ); }

}
