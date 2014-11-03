#include <SofaBaseCollision/IntrCapsuleOBB.inl>

namespace sofa{

using namespace defaulttype;

namespace component{

namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Vec3dTypes,Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Rigid3dTypes,Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Vec3fTypes,Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Rigid3fTypes,Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}

