

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

Creator<Contact::Factory, CompliantContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsuleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBCompliantContactClass("CompliantContact",true);

Creator<Contact::Factory, CompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > CylinderCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleCylinderCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types> > > OBBCylinderompliantContactClass("CompliantContact",true);


Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTriangleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleLineCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsuleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleRigidCapsuleCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSphereCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBCompliantContactClass("CompliantContact",true);
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleCylinderCompliantContactClass("CompliantContact",true);







////////////////////





Creator<Contact::Factory, CompliantContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >* PointPointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >* LineSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >* LinePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> >* LineLineCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >* TriangleSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >* TrianglePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> >* TriangleLineCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >* TriangleTriangleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >* SphereSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >* SpherePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> >* CapsuleCapsuleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >* CapsuleTriangleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >* CapsuleSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* OBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* SphereOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* CapsuleOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* TriangleOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<RigidSphereModel, RigidSphereModel> >* RigidSphereRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >* SphereRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >* LineRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >* TriangleRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidSpherePointCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >* CapsuleRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidSphereOBBCompliantContactClassNoStrip;

Creator<Contact::Factory, CompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* CylinderCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* SphereCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* CapsuleCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* TriangleCylinderCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* OBBCylinderompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* CylinderOBBompliantContactClassNoStrip;

Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidCapsuleTriangleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidCapsuleLineCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidCapsuleRigidCapsuleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> >* CapsuleRigidCapsuleCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidCapsuleSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> >* RigidCapsuleRigidSphereCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidCapsuleOBBCompliantContactClassNoStrip;
Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidCapsuleCylinderCompliantContactClassNoStrip;




void registerContactClasses()
{

	PointPointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	LineSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	LinePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	LineLineCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	TriangleSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	TrianglePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	TriangleLineCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	TriangleTriangleCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	SphereSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	SpherePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	CapsuleCapsuleCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	CapsuleTriangleCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	CapsuleSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	OBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	TriangleOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	SphereOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	CapsuleOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	RigidSphereRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<RigidSphereModel, RigidSphereModel> >("CompliantContact", true);
	SphereRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >("CompliantContact", true);
	LineRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >("CompliantContact", true);
	TriangleRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >("CompliantContact", true);
	RigidSpherePointCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> >("CompliantContact", true);
	CapsuleRigidSphereCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> >("CompliantContact", true);
	RigidSphereOBBCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);

	CylinderCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	SphereCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	CapsuleCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	TriangleCylinderCompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	OBBCylinderompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);
	CylinderOBBompliantContactClassNoStrip = new Creator<Contact::Factory, CompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >("CompliantContact", true);


    // TODO
    //Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidCapsuleTriangleCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidCapsuleLineCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidCapsuleRigidCapsuleCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> >* CapsuleRigidCapsuleCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >* RigidCapsuleSphereCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> >* RigidCapsuleRigidSphereCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidCapsuleOBBCompliantContactClassNoStrip;
//    Creator<Contact::Factory, CompliantContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >* RigidCapsuleCylinderCompliantContactClassNoStrip;


	
	TriangleOBBCompliantContactClassNoStrip->registerInFactory(); // Dummy function to avoid symbol stripping with some compilers
}

} // namespace collision

} // namespace component

} // namespace sofa
