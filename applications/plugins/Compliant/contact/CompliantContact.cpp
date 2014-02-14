

#include "CompliantContact.h"

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

  // TODO figure out what is this ? => ERROR ON WINDOWS !
//sofa::core::collision::DetectionOutput::ContactId Identifier::cpt=0;
//std::list<sofa::core::collision::DetectionOutput::ContactId> Identifier::availableId;

SOFA_DECL_CLASS(CompliantContact)


Creator<Contact::Factory, CompliantContact<PointModel, PointModel> > PointPointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, SphereModel> > LineSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, PointModel> > LinePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, LineModel> > LineLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, SphereModel> > TriangleSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, PointModel> > TrianglePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, LineModel> > TriangleLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, TriangleModel> > TriangleTriangleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, SphereModel> > SphereSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, PointModel> > SpherePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleModel, TriangleModel> > CapsuleTriangleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleModel, SphereModel> > CapsuleSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<OBBModel, OBBModel> > OBBOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, OBBModel> > SphereOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleModel, OBBModel> > CapsuleOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, OBBModel> > TriangleOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, RigidSphereModel> > SphereRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineModel, RigidSphereModel> > LineRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<RigidSphereModel, PointModel> > RigidSpherePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBCompliantContactClass("CompliantContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
