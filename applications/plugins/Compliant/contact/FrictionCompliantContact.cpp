#include "FrictionCompliantContact.h"

#include <sofa/component/collision/FrictionContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;

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


} // namespace collision

} // namespace component

} // namespace sofa

