#include <SofaMeshCollision/IntrTriangleOBB.inl>

namespace sofa{
namespace component{
namespace collision{

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
#ifndef SOFA_FLOAT
template class SOFA_MESH_COLLISION_API TIntrTriangleOBB<defaulttype::Vec3dTypes,defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MESH_COLLISION_API TIntrTriangleOBB<defaulttype::Vec3fTypes,defaulttype::Rigid3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
