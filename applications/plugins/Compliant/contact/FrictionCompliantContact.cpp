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


sofa::core::collision::ContactCreator< FrictionCompliantContact<PointModel, PointModel> > PointPointFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<LineModel, SphereModel> > LineSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<LineModel, PointModel> > LinePointFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<LineModel, LineModel> > LineLineFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, SphereModel> > TriangleSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, PointModel> > TrianglePointFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, LineModel> > TriangleLineFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, TriangleModel> > TriangleTriangleFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<SphereModel, SphereModel> > SphereSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<SphereModel, PointModel> > SpherePointFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsuleFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CapsuleModel, TriangleModel> > CapsuleTriangleFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CapsuleModel, SphereModel> > CapsuleSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<OBBModel, OBBModel> > OBBOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<SphereModel, OBBModel> > SphereOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CapsuleModel, OBBModel> > CapsuleOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, OBBModel> > TriangleOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<SphereModel, RigidSphereModel> > SphereRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<LineModel, RigidSphereModel> > LineRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<RigidSphereModel, PointModel> > RigidSpherePointFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<OBBModel ,RigidSphereModel> > OBBRigidSphereFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CylinderModel, OBBModel> >	CylinderModelOBBModelFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CylinderModel, TriangleModel> >	CylinderModelTriangleModelFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<TriangleModel, CylinderModel> >	TriangleModelCylinderModelFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<CylinderModel, SphereModel> >	CylinderModelSphereModelFrictionCompliantContactClass("FrictionCompliantContact",true);
sofa::core::collision::ContactCreator< FrictionCompliantContact<SphereModel, CylinderModel> >	SphereModelCylinderModelFrictionCompliantContactClass("FrictionCompliantContact",true);

} // namespace collision

} // namespace component

} // namespace sofa

