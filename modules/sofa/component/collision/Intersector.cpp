#include <sofa/component/collision/Intersector.inl>
#include <sofa/component/component.h>

namespace sofa{

namespace component{

namespace collision{

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API Intersector<Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API Intersector<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API Intersector<Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API Intersector<Vec3fTypes>;
#endif
//----------------------------------------------------------------------------

}
}
}
