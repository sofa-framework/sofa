#include <sofa/component/collision/IntrMeshUtility.inl>

namespace sofa{
namespace component{
namespace collision{

#ifndef SOFA_FLOAT
template class SOFA_MESH_COLLISION_API IntrUtil<TTriangle<Vec3dTypes> >;
template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<Vec3dTypes>,TOBB<Rigid3dTypes> >;
template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
template class SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<Vec3dTypes> >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MESH_COLLISION_API IntrUtil<TTriangle<Vec3fTypes> >;
template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<Vec3fTypes>,TOBB<Rigid3fTypes> >;
template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
template class SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<Vec3fTypes> >;
#endif


}
}
}
