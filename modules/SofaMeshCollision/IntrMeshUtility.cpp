#include <SofaMeshCollision/IntrMeshUtility.inl>

namespace sofa{
namespace component{
namespace collision{

#ifndef SOFA_FLOAT
template struct SOFA_MESH_COLLISION_API IntrUtil<TTriangle<defaulttype::Vec3dTypes> >;
template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<defaulttype::Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<defaulttype::Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
template struct SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<defaulttype::Vec3dTypes> >;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_MESH_COLLISION_API IntrUtil<TTriangle<defaulttype::Vec3fTypes> >;
template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<defaulttype::Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<defaulttype::Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
template struct SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<defaulttype::Vec3fTypes> >;
#endif


}
}
}
