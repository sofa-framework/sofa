#include <sofa/component/collision/IntrCapsuleOBB.inl>

namespace sofa{

namespace component{

namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Vec3dTypes,Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TIntrCapsuleOBB<Vec3fTypes,Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}

