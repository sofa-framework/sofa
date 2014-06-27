#include <SofaBaseCollision/IntrSphereOBB.inl>


namespace sofa{
using namespace defaulttype;
namespace component{
namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Vec3dTypes,Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Rigid3dTypes,Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Vec3fTypes,Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TIntrSphereOBB<Rigid3fTypes,Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
