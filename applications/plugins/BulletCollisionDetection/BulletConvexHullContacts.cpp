#include "BulletConvexHullModel.h"
#include <sofa/component/collision/BarycentricPenalityContact.inl>
#include <sofa/component/collision/BarycentricContactMapper.inl>
#include "BulletConvexHullContactMapper.h"
#include <sofa/component/collision/FrictionContact.h>
#include <sofa/component/collision/BarycentricDistanceLMConstraintContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{


using namespace defaulttype;
using simulation::Node;

Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, BulletConvexHullModel> > BulletConvexHullModelBulletConvexHullContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, SphereModel> > BulletConvexHullModelSphereModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, RigidSphereModel> > BulletConvexHullModelRigidSphereModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, TriangleModel> > BulletConvexHullModelTriangleModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, LineModel> > BulletConvexHullModelLineModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, PointModel> > BulletConvexHullModelPointModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, OBBModel> > BulletConvexHullModelOBBModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CapsuleModel> > BulletConvexHullModelCapsuleModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, RigidCapsuleModel> > BulletConvexHullModelRigidCapsuleModelContactClass("default",true);
Creator<sofa::core::collision::Contact::Factory, BarycentricPenalityContact<BulletConvexHullModel, CylinderModel> > BulletConvexHullModelCylinderModelContactClass("default",true);


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
