

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

SOFA_DECL_CLASS(PenaltyCompliantContact)


Creator<Contact::Factory, PenaltyCompliantContact<PointModel, PointModel> > PointPointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineModel, SphereModel> > LineSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineModel, PointModel> > LinePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineModel, LineModel> > LineLinePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleModel, SphereModel> > TriangleSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleModel, PointModel> > TrianglePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleModel, LineModel> > TriangleLinePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleModel, TriangleModel> > TriangleTrianglePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereModel, SphereModel> > SphereSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereModel, PointModel> > SpherePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsulePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleModel, TriangleModel> > CapsuleTrianglePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleModel, SphereModel> > CapsuleSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<OBBModel, OBBModel> > OBBOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereModel, OBBModel> > SphereOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleModel, OBBModel> > CapsuleOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleModel, OBBModel> > TriangleOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<SphereModel, RigidSphereModel> > SphereRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<LineModel, RigidSphereModel> > LineRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<RigidSphereModel, PointModel> > RigidSpherePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
Creator<Contact::Factory, PenaltyCompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
