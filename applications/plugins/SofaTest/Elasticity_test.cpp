#include "Elasticity_test.inl"
namespace sofa {

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template class SOFA_TestPlugin_API Elasticity_test<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_TestPlugin_API Elasticity_test<Vec3fTypes>;
#endif
}
