#include "Elasticity_test.inl"
namespace sofa {

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template struct SOFA_SOFATEST_API Elasticity_test<sofa::defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_SOFATEST_API Elasticity_test<sofa::defaulttype::Vec3fTypes>;
#endif
}
