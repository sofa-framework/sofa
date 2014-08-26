

#include "PenalityCompliantContact.h"

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

SOFA_DECL_CLASS(PenalityCompliantContact)


Creator<Contact::Factory, PenalityCompliantContact<PointModel, PointModel> > PointPointPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<LineModel, SphereModel> > LineSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<LineModel, PointModel> > LinePointPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<LineModel, LineModel> > LineLinePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<TriangleModel, SphereModel> > TriangleSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<TriangleModel, PointModel> > TrianglePointPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<TriangleModel, LineModel> > TriangleLinePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<TriangleModel, TriangleModel> > TriangleTrianglePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<SphereModel, SphereModel> > SphereSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<SphereModel, PointModel> > SpherePointPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsulePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<CapsuleModel, TriangleModel> > CapsuleTrianglePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<CapsuleModel, SphereModel> > CapsuleSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<OBBModel, OBBModel> > OBBOBBPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<SphereModel, OBBModel> > SphereOBBPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<CapsuleModel, OBBModel> > CapsuleOBBPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<TriangleModel, OBBModel> > TriangleOBBPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<SphereModel, RigidSphereModel> > SphereRigidSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<LineModel, RigidSphereModel> > LineRigidSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<RigidSphereModel, PointModel> > RigidSpherePointPenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSpherePenalityCompliantContactClass("PenalityCompliantContact",true);
Creator<Contact::Factory, PenalityCompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBPenalityCompliantContactClass("PenalityCompliantContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
