

#include "PenaltyCompliantContact.h"

#include <SofaConstraint/FrictionContact.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>


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

Creator<Contact::Factory, PenaltyCompliantContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLinePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLinePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTrianglePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsulePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTrianglePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
