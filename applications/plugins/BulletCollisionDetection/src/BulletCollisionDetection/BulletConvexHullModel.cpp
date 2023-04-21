#include "BulletConvexHullModel.inl"

namespace sofa
{

namespace component
{

namespace collision
{

int BulletConvexHullModelClass = core::RegisterObject("collision model using a set of convex hulls")
        .add< TBulletConvexHullModel< defaulttype::Rigid3Types> >()

        .addAlias("BulletConvexHullModel")
        .addAlias("BtConvexHullModel")
        .addAlias("BulletConvexHull")
        .addAlias("BtConvexHull")
        ;


template class SOFA_BULLETCOLLISIONDETECTION_API TBulletConvexHullModel<defaulttype::Rigid3Types>;


}
}
}
