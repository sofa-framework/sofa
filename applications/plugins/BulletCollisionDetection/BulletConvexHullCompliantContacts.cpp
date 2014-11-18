
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
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, CylinderModel> > BCHullCylinderCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, TriangleModel> > BCHullTriangleCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, RigidCapsuleModel> > BCHullRigidCapsuleCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, CapsuleModel> > CapsuleBCHullCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, SphereModel> > BCHullSphereCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereCompliantContactClassClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<BulletConvexHullModel, OBBModel> > BCHullOBBCompliantContactClassClass("CompliantContact",true);

Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullFrictionCompliantContactClassClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, CylinderModel> > BCHullCylinderFrictionCompliantContactClassClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, TriangleModel> > BCHullTriangleFrictionCompliantContactClassClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, RigidCapsuleModel> > BCHullRigidCapsuleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, CapsuleModel> > CapsuleBCHullFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, SphereModel> > BCHullSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<BulletConvexHullModel, OBBModel> > BCHullOBBCompliantFrictionCompliantContactClass("FrictionCompliantContact",true);
} // namespace collision

} // namespace component

} // namespace sofa

