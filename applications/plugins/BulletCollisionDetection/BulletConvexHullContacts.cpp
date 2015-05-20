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

sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, BulletConvexHullModel> > BulletConvexHullModelBulletConvexHullContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, SphereModel> > BulletConvexHullModelSphereModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, RigidSphereModel> > BulletConvexHullModelRigidSphereModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, TriangleModel> > BulletConvexHullModelTriangleModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, LineModel> > BulletConvexHullModelLineModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, PointModel> > BulletConvexHullModelPointModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, OBBModel> > BulletConvexHullModelOBBModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, CapsuleModel> > BulletConvexHullModelCapsuleModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, RigidCapsuleModel> > BulletConvexHullModelRigidCapsuleModelContactClass("default",true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<BulletConvexHullModel, CylinderModel> > BulletConvexHullModelCylinderModelContactClass("default",true);


sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, BulletConvexHullModel> > BCHullBCHullLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CylinderModel> > BCHullCylinderLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, TriangleModel> > BCHullTriangleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, RigidCapsuleModel> > BCHullRigidCapsuleLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, CapsuleModel> > CapsuleBCHullLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, SphereModel> > BCHullSphereLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, RigidSphereModel> > BCHullRigidSphereLMConstraintContactClassClass("distanceLMConstraint",true);
sofa::core::collision::ContactCreator< BarycentricDistanceLMConstraintContact<BulletConvexHullModel, OBBModel> > BCHullOBBLMConstraintContactClassClass("distanceLMConstraint",true);

}
}
}
