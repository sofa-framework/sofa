#include "BulletConvexHullModel.h"
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include "BulletConvexHullContactMapper.h"
#include <SofaConstraint/FrictionContact.h>
#include <SofaConstraint/BarycentricDistanceLMConstraintContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{


using namespace defaulttype;
using simulation::Node;
using core::collision::Contact;

Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, BulletConvexHullModel> > BulletConvexHullModelBulletConvexHullContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, SphereModel> > BulletConvexHullModelSphereModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, RigidSphereModel> > BulletConvexHullModelRigidSphereModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, TriangleModel> > BulletConvexHullModelTriangleModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, LineModel> > BulletConvexHullModelLineModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, PointModel> > BulletConvexHullModelPointModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, OBBModel> > BulletConvexHullModelOBBModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CapsuleModel> > BulletConvexHullModelCapsuleModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, RigidCapsuleModel> > BulletConvexHullModelRigidCapsuleModelContactClass("default",true);
Creator<Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CylinderModel> > BulletConvexHullModelCylinderModelContactClass("default",true);


Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CylinderModel> > BCHullCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, TriangleModel> > BCHullTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, RigidCapsuleModel> > BCHullRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CapsuleModel> > CapsuleBCHullLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, SphereModel> > BCHullSphereLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<BulletConvexHullModel, OBBModel> > BCHullOBBLMConstraintContactClassClass("distanceLMConstraint",true);

}
}
}
