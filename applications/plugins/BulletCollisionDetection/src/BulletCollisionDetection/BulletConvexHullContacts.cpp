#include "BulletConvexHullModel.h"

#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>
#include <sofa/component/collision/response/contact/BarycentricPenalityContact.inl>
#include <sofa/component/collision/response/contact/FrictionContact.inl>

#include <CollisionOBBCapsule/response/mapper/CapsuleContactMapper.h>
#include <CollisionOBBCapsule/response/mapper/OBBContactMapper.h>

#include "BulletConvexHullContactMapper.h"

#include <CollisionOBBCapsule/geometry/OBBModel.h>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>

using namespace sofa::defaulttype;
using sofa::simulation::Node;
using sofa::core::collision::Contact;
using namespace sofa::component::collision::geometry;
using namespace collisionobbcapsule::geometry;
using namespace sofa::component::collision::response::contact;

namespace sofa
{

namespace component
{

namespace collision
{



Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, BulletConvexHullModel> > BulletConvexHullModelBulletConvexHullContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > BulletConvexHullModelSphereModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, RigidSphereModel> > BulletConvexHullModelRigidSphereModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > BulletConvexHullModelTriangleModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > BulletConvexHullModelLineModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > BulletConvexHullModelPointModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > BulletConvexHullModelOBBModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > BulletConvexHullModelCapsuleModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > BulletConvexHullModelRigidCapsuleModelContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > BulletConvexHullModelCylinderModelContactClass("PenalityContactForceField",true);


//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > BCHullTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleBCHullLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > BCHullSphereLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
//Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > BCHullOBBLMConstraintContactClassClass("distanceLMConstraint",true);

}
}
}
