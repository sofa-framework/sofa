
#include "BulletConvexHullModel.h"
#include <Compliant/contact/CompliantContact.h>
#include <Compliant/contact/FrictionCompliantContact.h>


#include <SofaConstraint/FrictionContact.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include "BulletConvexHullContactMapper.h"


namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;
using core::collision::Contact;

sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, CylinderModel> > BCHullCylinderCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, TriangleModel> > BCHullTriangleCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, RigidCapsuleModel> > BCHullRigidCapsuleCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, CapsuleModel> > CapsuleBCHullCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, SphereModel> > BCHullSphereCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereCompliantContactClassClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<BulletConvexHullModel, OBBModel> > BCHullOBBCompliantContactClassClass("CompliantContact",true);

sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullFrictionCompliantContactClassClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, CylinderModel> > BCHullCylinderFrictionCompliantContactClassClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, TriangleModel> > BCHullTriangleFrictionCompliantContactClassClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, RigidCapsuleModel> > BCHullRigidCapsuleFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, CapsuleModel> > CapsuleBCHullFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, SphereModel> > BCHullSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<BulletConvexHullModel, OBBModel> > BCHullOBBCompliantFrictionCompliantContactClass("FrictionCompliantContact",true);
} // namespace collision

} // namespace component

} // namespace sofa

