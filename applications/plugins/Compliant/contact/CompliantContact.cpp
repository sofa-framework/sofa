

#include "CompliantContact.h"

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

Creator<Contact::Factory, CompliantContact<CylinderModel, CylinderModel> > CylinderCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereModel, CylinderModel> > SphereCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleModel, CylinderModel> > CapsuleCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleModel, CylinderModel> > TriangleCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CylinderModel, OBBModel > > OBBCylinderompliantContactClass("CompliantContact",true);

Creator<Contact::Factory, CompliantContact<PointModel, PointModel> >* PointPointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineModel, SphereModel> >* LineSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineModel, PointModel> >* LinePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineModel, LineModel> >* LineLineCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, SphereModel> >* TriangleSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, PointModel> >* TrianglePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, LineModel> >* TriangleLineCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, TriangleModel> >* TriangleTriangleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereModel, SphereModel> >* SphereSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereModel, PointModel> >* SpherePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleModel, CapsuleModel> >* CapsuleCapsuleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleModel, TriangleModel> >* CapsuleTriangleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleModel, SphereModel> >* CapsuleSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<OBBModel, OBBModel> >* OBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereModel, OBBModel> >* SphereOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleModel, OBBModel> >* CapsuleOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, OBBModel> >* TriangleOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<RigidSphereModel, RigidSphereModel> >* RigidSphereRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereModel, RigidSphereModel> >* SphereRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineModel, RigidSphereModel> >* LineRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, RigidSphereModel> >* TriangleRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<RigidSphereModel, PointModel> >* RigidSpherePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleModel, RigidSphereModel> >* CapsuleRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<RigidSphereModel, OBBModel> >* RigidSphereOBBCompliantContactClassNoStrip;

Creator<Contact::Factory, CompliantContact<CylinderModel, CylinderModel> >* CylinderCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereModel, CylinderModel> >* SphereCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleModel, CylinderModel> >* CapsuleCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleModel, CylinderModel> >* TriangleCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<OBBModel, CylinderModel> >* OBBCylinderompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CylinderModel, OBBModel> >* CylinderOBBompliantContactClassNoStrip;


void registerContactClasses()
{

	PointPointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<PointModel, PointModel> >("CompliantContact", true);
	LineSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineModel, SphereModel> >("CompliantContact", true);
	LinePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineModel, PointModel> >("CompliantContact", true);
	LineLineCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineModel, LineModel> >("CompliantContact", true);
	TriangleSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, SphereModel> >("CompliantContact", true);
	TrianglePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, PointModel> >("CompliantContact", true);
	TriangleLineCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, LineModel> >("CompliantContact", true);
	TriangleTriangleCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, TriangleModel> >("CompliantContact", true);
	SphereSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereModel, SphereModel> >("CompliantContact", true);
	SpherePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereModel, PointModel> >("CompliantContact", true);
	CapsuleCapsuleCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleModel, CapsuleModel> >("CompliantContact", true);
	CapsuleTriangleCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleModel, TriangleModel> >("CompliantContact", true);
	CapsuleSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleModel, SphereModel> >("CompliantContact", true);
	OBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<OBBModel, OBBModel> >("CompliantContact", true);
	TriangleOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, OBBModel> >("CompliantContact", true);
	SphereOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereModel, OBBModel> >("CompliantContact", true);
	CapsuleOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleModel, OBBModel> >("CompliantContact", true);
	RigidSphereRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<RigidSphereModel, RigidSphereModel> >("CompliantContact", true);
	SphereRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereModel, RigidSphereModel> >("CompliantContact", true);
	LineRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineModel, RigidSphereModel> >("CompliantContact", true);
	TriangleRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, RigidSphereModel> >("CompliantContact", true);
	RigidSpherePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<RigidSphereModel, PointModel> >("CompliantContact", true);
	CapsuleRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleModel, RigidSphereModel> >("CompliantContact", true);
	RigidSphereOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<RigidSphereModel, OBBModel> >("CompliantContact", true);

	CylinderCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CylinderModel, CylinderModel> >("CompliantContact", true);
	SphereCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereModel, CylinderModel> >("CompliantContact", true);
	CapsuleCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleModel, CylinderModel> >("CompliantContact", true);
	TriangleCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleModel, CylinderModel> >("CompliantContact", true);
	OBBCylinderompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<OBBModel, CylinderModel> >("CompliantContact", true);

	CylinderOBBompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CylinderModel, OBBModel> >("CompliantContact", true);
	
	TriangleOBBCompliantContactClassNoStrip->registerInFactory(); // Dummy function to avoid symbol stripping with some compilers
}

} // namespace collision

} // namespace component

} // namespace sofa
