#include <SofaBaseCollision/IntrOBBOBB.inl>

namespace sofa{

namespace component{

namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TIntrOBBOBB<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TIntrOBBOBB<defaulttype::Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
