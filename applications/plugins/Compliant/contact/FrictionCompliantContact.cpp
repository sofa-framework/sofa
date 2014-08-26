#include "FrictionCompliantContact.h"

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

SOFA_DECL_CLASS(FrictionCompliantContact)


Creator<Contact::Factory, FrictionCompliantContact<PointModel, PointModel> > PointPointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineModel, SphereModel> > LineSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineModel, PointModel> > LinePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineModel, LineModel> > LineLineFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, SphereModel> > TriangleSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, PointModel> > TrianglePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, LineModel> > TriangleLineFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, TriangleModel> > TriangleTriangleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereModel, SphereModel> > SphereSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereModel, PointModel> > SpherePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleModel, TriangleModel> > CapsuleTriangleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleModel, SphereModel> > CapsuleSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<OBBModel, OBBModel> > OBBOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereModel, OBBModel> > SphereOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleModel, OBBModel> > CapsuleOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, OBBModel> > TriangleOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereModel, RigidSphereModel> > SphereRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineModel, RigidSphereModel> > LineRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<RigidSphereModel, PointModel> > RigidSpherePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<OBBModel ,RigidSphereModel> > OBBRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CylinderModel, OBBModel> >	CylinderModelOBBModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CylinderModel, TriangleModel> >	CylinderModelTriangleModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleModel, CylinderModel> >	TriangleModelCylinderModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CylinderModel, SphereModel> >	CylinderModelSphereModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereModel, CylinderModel> >	SphereModelCylinderModelFrictionCompliantContactClass("FrictionCompliantContact",true);

} // namespace collision

} // namespace component

} // namespace sofa

