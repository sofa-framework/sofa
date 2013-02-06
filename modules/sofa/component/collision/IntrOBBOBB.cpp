#include <sofa/component/collision/IntrOBBOBB.inl>

namespace sofa{

namespace component{

namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template
class SOFA_BASE_COLLISION_API TIntrOBBOBB<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template
class SOFA_BASE_COLLISION_API TIntrOBBOBB<Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
