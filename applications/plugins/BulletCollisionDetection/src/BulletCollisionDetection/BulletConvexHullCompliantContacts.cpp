
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

Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullCylinderCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > BCHullTriangleCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullRigidCapsuleCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleBCHullCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > BCHullSphereCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullOBBCompliantContactClassClass("CompliantContact",true);

Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullFrictionCompliantContactClassClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullCylinderFrictionCompliantContactClassClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > BCHullTriangleFrictionCompliantContactClassClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullRigidCapsuleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleBCHullFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > BCHullSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullOBBCompliantFrictionCompliantContactClass("FrictionCompliantContact",true);
} // namespace collision

} // namespace component

} // namespace sofa

