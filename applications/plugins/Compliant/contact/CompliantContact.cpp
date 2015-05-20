

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


sofa::core::collision::ContactCreator< CompliantContact<PointModel, PointModel> > PointPointCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<LineModel, SphereModel> > LineSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<LineModel, PointModel> > LinePointCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<LineModel, LineModel> > LineLineCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, SphereModel> > TriangleSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, PointModel> > TrianglePointCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, LineModel> > TriangleLineCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, TriangleModel> > TriangleTriangleCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, SphereModel> > SphereSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, PointModel> > SpherePointCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, TriangleModel> > CapsuleTriangleCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, SphereModel> > CapsuleSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<OBBModel, OBBModel> > OBBOBBCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, OBBModel> > SphereOBBCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, OBBModel> > CapsuleOBBCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, OBBModel> > TriangleOBBCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, RigidSphereModel> > SphereRigidSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<LineModel, RigidSphereModel> > LineRigidSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, PointModel> > RigidSpherePointCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBCompliantContactClass("CompliantContact",true);

sofa::core::collision::ContactCreator< CompliantContact<CylinderModel, CylinderModel> > CylinderCylinderCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, CylinderModel> > SphereCylinderCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, CylinderModel> > CapsuleCylinderCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, CylinderModel> > TriangleCylinderCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<CylinderModel, OBBModel > > OBBCylinderompliantContactClass("CompliantContact",true);


sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, TriangleModel> > RigidCapsuleTriangleCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, LineModel> > RigidCapsuleLineCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, RigidCapsuleModel> > RigidCapsuleRigidCapsuleCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, CapsuleModel> > CapsuleRigidCapsuleCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, SphereModel> > RigidCapsuleSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, RigidSphereModel> > RigidCapsuleRigidSphereCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, OBBModel> > RigidCapsuleOBBCompliantContactClass("CompliantContact",true);
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, CylinderModel> > RigidCapsuleCylinderCompliantContactClass("CompliantContact",true);







////////////////////






sofa::core::collision::ContactCreator< CompliantContact<PointModel, PointModel> >* PointPointCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<LineModel, SphereModel> >* LineSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<LineModel, PointModel> >* LinePointCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<LineModel, LineModel> >* LineLineCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, SphereModel> >* TriangleSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, PointModel> >* TrianglePointCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, LineModel> >* TriangleLineCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, TriangleModel> >* TriangleTriangleCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, SphereModel> >* SphereSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, PointModel> >* SpherePointCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, CapsuleModel> >* CapsuleCapsuleCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, TriangleModel> >* CapsuleTriangleCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, SphereModel> >* CapsuleSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<OBBModel, OBBModel> >* OBBCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, OBBModel> >* SphereOBBCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, OBBModel> >* CapsuleOBBCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, OBBModel> >* TriangleOBBCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, RigidSphereModel> >* RigidSphereRigidSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, RigidSphereModel> >* SphereRigidSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<LineModel, RigidSphereModel> >* LineRigidSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, RigidSphereModel> >* TriangleRigidSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, PointModel> >* RigidSpherePointCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, RigidSphereModel> >* CapsuleRigidSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, OBBModel> >* RigidSphereOBBCompliantContactClassNoStrip;

sofa::core::collision::ContactCreator< CompliantContact<CylinderModel, CylinderModel> >* CylinderCylinderCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<SphereModel, CylinderModel> >* SphereCylinderCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, CylinderModel> >* CapsuleCylinderCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, CylinderModel> >* TriangleCylinderCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<OBBModel, CylinderModel> >* OBBCylinderompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<CylinderModel, OBBModel> >* CylinderOBBompliantContactClassNoStrip;

sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, TriangleModel> >* RigidCapsuleTriangleCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, LineModel> >* RigidCapsuleLineCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, RigidCapsuleModel> >* RigidCapsuleRigidCapsuleCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, CapsuleModel> >* CapsuleRigidCapsuleCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, SphereModel> >* RigidCapsuleSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, RigidSphereModel> >* RigidCapsuleRigidSphereCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, OBBModel> >* RigidCapsuleOBBCompliantContactClassNoStrip;
sofa::core::collision::ContactCreator< CompliantContact<RigidCapsuleModel, CylinderModel> >* RigidCapsuleCylinderCompliantContactClassNoStrip;




void registerContactClasses()
{

	PointPointCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<PointModel, PointModel> >("CompliantContact", true);
	LineSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<LineModel, SphereModel> >("CompliantContact", true);
	LinePointCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<LineModel, PointModel> >("CompliantContact", true);
	LineLineCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<LineModel, LineModel> >("CompliantContact", true);
	TriangleSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, SphereModel> >("CompliantContact", true);
	TrianglePointCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, PointModel> >("CompliantContact", true);
	TriangleLineCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, LineModel> >("CompliantContact", true);
	TriangleTriangleCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, TriangleModel> >("CompliantContact", true);
	SphereSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<SphereModel, SphereModel> >("CompliantContact", true);
	SpherePointCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<SphereModel, PointModel> >("CompliantContact", true);
	CapsuleCapsuleCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, CapsuleModel> >("CompliantContact", true);
	CapsuleTriangleCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, TriangleModel> >("CompliantContact", true);
	CapsuleSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, SphereModel> >("CompliantContact", true);
	OBBCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<OBBModel, OBBModel> >("CompliantContact", true);
	TriangleOBBCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, OBBModel> >("CompliantContact", true);
	SphereOBBCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<SphereModel, OBBModel> >("CompliantContact", true);
	CapsuleOBBCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, OBBModel> >("CompliantContact", true);
	RigidSphereRigidSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, RigidSphereModel> >("CompliantContact", true);
	SphereRigidSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<SphereModel, RigidSphereModel> >("CompliantContact", true);
	LineRigidSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<LineModel, RigidSphereModel> >("CompliantContact", true);
	TriangleRigidSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, RigidSphereModel> >("CompliantContact", true);
	RigidSpherePointCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, PointModel> >("CompliantContact", true);
	CapsuleRigidSphereCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, RigidSphereModel> >("CompliantContact", true);
	RigidSphereOBBCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<RigidSphereModel, OBBModel> >("CompliantContact", true);

	CylinderCylinderCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CylinderModel, CylinderModel> >("CompliantContact", true);
	SphereCylinderCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<SphereModel, CylinderModel> >("CompliantContact", true);
	CapsuleCylinderCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CapsuleModel, CylinderModel> >("CompliantContact", true);
	TriangleCylinderCompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<TriangleModel, CylinderModel> >("CompliantContact", true);
	OBBCylinderompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<OBBModel, CylinderModel> >("CompliantContact", true);

	CylinderOBBompliantContactClassNoStrip = new sofa::core::collision::ContactCreator< CompliantContact<CylinderModel, OBBModel> >("CompliantContact", true);


    // TODO
    //Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, TriangleModel> >* RigidCapsuleTriangleCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, LineModel> >* RigidCapsuleLineCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, RigidCapsuleModel> >* RigidCapsuleRigidCapsuleCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, CapsuleModel> >* CapsuleRigidCapsuleCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, SphereModel> >* RigidCapsuleSphereCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, RigidSphereModel> >* RigidCapsuleRigidSphereCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, OBBModel> >* RigidCapsuleOBBCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<RigidCapsuleModel, CylinderModel> >* RigidCapsuleCylinderCompliantContactClassNoStrip;


	
	TriangleOBBCompliantContactClassNoStrip->registerInFactory(); // Dummy function to avoid symbol stripping with some compilers
}

} // namespace collision

} // namespace component

} // namespace sofa
