#include "FrictionCompliantContact.h"

#include <SofaConstraint/FrictionContact.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaMiscCollision/CapsuleContactMapper.h>
#include <SofaMiscCollision/OBBContactMapper.h>

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

Creator<Contact::Factory, FrictionCompliantContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsuleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types> ,RigidSphereModel> > OBBRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> >	CylinderModelOBBModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> >	CylinderModelTriangleModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >	TriangleModelCylinderModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> >	CylinderModelSphereModelFrictionCompliantContactClass("FrictionCompliantContact",true);
Creator<Contact::Factory, FrictionCompliantContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> >	SphereModelCylinderModelFrictionCompliantContactClass("FrictionCompliantContact",true);

} // namespace collision

} // namespace component

} // namespace sofa

