#include <sofa/component/collision/Intersector.inl>
#include <component.h>

namespace sofa{

namespace component{

namespace collision{


//----------------------------------------------------------------------------
// Explicit instantiation.
//----------------------------------------------------------------------------
// template WM5_MATHEMATICS_ITEM
// class Intersector<float,Vector2f>;

//template class Intersector<float,Vec<3,float> >;

// template WM5_MATHEMATICS_ITEM
// class Intersector<double,Vector2d>;

//template class Intersector<double,Vec<3,double> >;

#ifndef SOFA_FLOAT
template
class SOFA_BASE_COLLISION_API Intersector<Rigid3dTypes>;
class SOFA_BASE_COLLISION_API Intersector<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template
class SOFA_BASE_COLLISION_API Intersector<Rigid3fTypes>;
class SOFA_BASE_COLLISION_API Intersector<Vec3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
