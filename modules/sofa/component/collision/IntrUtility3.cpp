#include <sofa/component/collision/IntrUtility3.inl>

namespace sofa{
namespace component{
namespace collision{

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
template struct SOFA_BASE_COLLISION_API IntrUtil<double>;

#ifndef SOFA_FLOAT

template class SOFA_BASE_COLLISION_API IntrConfiguration<double>;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<double>;
template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane<double> (const Vec<3,double>&, double,int&, Vec<3,double>*);
template SOFA_BASE_COLLISION_API Vec<3,double> GetPointFromIndex<double> (int, const MyBox<double>&);
template SOFA_BASE_COLLISION_API class CapIntrConfiguration<double>;
#ifndef SOFA_FLAG_SOFAPRO
template struct SOFA_BASE_COLLISION_API IntrUtil<TOBB<RigidTypes> >;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<TOBB<Rigid3dTypes> >;
template class SOFA_BASE_COLLISION_API IntrAxis<TOBB<Rigid3dTypes> >;
template class SOFA_BASE_COLLISION_API FindContactSet<TOBB<Rigid3dTypes> >;
template SOFA_BASE_COLLISION_API Vec<3,Rigid3dTypes::Real> getPointFromIndex<Rigid3dTypes> (int index, const TOBB<Rigid3dTypes>& box);
#endif // SOFA_FLAG_SOFAPRO
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_BASE_COLLISION_API IntrUtil<float>;
template class SOFA_BASE_COLLISION_API IntrConfiguration<float>;

template SOFA_BASE_COLLISION_API void ClipConvexPolygonAgainstPlane<float> (const Vec<3,float>&, float,int&, Vec<3,float>*);
template SOFA_BASE_COLLISION_API Vec<3,float> GetPointFromIndex<float> (int index, const MyBox<float>& box);
template SOFA_BASE_COLLISION_API class CapIntrConfiguration<float>;

#ifndef SOFA_FLAG_SOFAPRO
template struct SOFA_BASE_COLLISION_API IntrUtil<TOBB<Rigid3fTypes> >;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<TOBB<Rigid3fTypes> >;
template struct SOFA_BASE_COLLISION_API IntrConfigManager<float>;
template class SOFA_BASE_COLLISION_API IntrAxis<TOBB<Rigid3fTypes> >;
template class SOFA_BASE_COLLISION_API FindContactSet<TOBB<Rigid3fTypes> >;
template SOFA_BASE_COLLISION_API Vec<3,Rigid3fTypes::Real> getPointFromIndex<Rigid3fTypes> (int index, const TOBB<Rigid3fTypes>& box);
#endif // SOFA_FLAG_SOFAPRO
#endif
//----------------------------------------------------------------------------

}
}
}

