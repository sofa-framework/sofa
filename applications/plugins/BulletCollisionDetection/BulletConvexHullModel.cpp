#include "BulletConvexHullModel.inl"

namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(BulletConvexHull)

int BulletConvexHullModelClass = core::RegisterObject("collision model using a set of convex hulls")
#ifndef SOFA_FLOAT
        .add< TBulletConvexHullModel< defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletConvexHullModel< defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("BulletConvexHullModel")
        .addAlias("BtConvexHullModel")
        .addAlias("BulletConvexHull")
        .addAlias("BtConvexHull")
        ;


#ifndef SOFA_FLOAT
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletConvexHullModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletConvexHullModel<defaulttype::Rigid3fTypes>;
#endif

}
}
}
