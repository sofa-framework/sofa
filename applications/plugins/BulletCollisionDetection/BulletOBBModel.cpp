#include "BulletOBBModel.inl"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

int BulletOBBModelClass = core::RegisterObject("collision model using a set of OBBs, it can be used in the bullet collision pipeline")
#ifndef SOFA_FLOAT
        .add< TBulletOBBModel<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TBulletOBBModel<defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("BulletOBBModel")
        .addAlias("BtOBBModel")
        .addAlias("BulletOBB")
        .addAlias("BtOBB")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletOBBModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BULLETCOLLISIONDETECTION_API TBulletOBBModel<defaulttype::Rigid3fTypes>;
#endif


//SofaBox::SofaBox( const btVector3& boxHalfExtents)
//: btBoxShape(boxHalfExtents)
//{
//}

//void SofaBox::getAabb(const btTransform &t, btVector3 &aabbMin, btVector3 &aabbMax) const{
//    const btVector3 & halfExtentsWithMargin = getHalfExtentsWithMargin();
//    btMatrix3x3 abs_b = t.getBasis().absolute();
//    btVector3 center = t.getOrigin();
//    btVector3 extent = halfExtentsWithMargin.dot3( abs_b[0], abs_b[1], abs_b[2] );
//    aabbMin = center - extent;
//    aabbMax = center + extent;
//}

} // namespace collision

} // namespace component

} // namespace sofa
