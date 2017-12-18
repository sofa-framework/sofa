#include "Elasticity_test_deprecated.inl"
namespace sofa {

using namespace sofa::defaulttype;

#ifdef SOFA_WITH_DOUBLE
template struct SOFA_SOFATEST_API Elasticity_test_deprecated<sofa::defaulttype::Vec3dTypes>;
#endif
#ifdef SOFA_WITH_FLOAT
template struct SOFA_SOFATEST_API Elasticity_test_deprecated<sofa::defaulttype::Vec3fTypes>;
#endif
}
