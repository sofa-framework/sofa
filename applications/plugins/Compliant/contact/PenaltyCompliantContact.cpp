

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


sofa::core::collision::ContactCreator< PenaltyCompliantContact<PointModel, PointModel> > PointPointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<LineModel, SphereModel> > LineSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<LineModel, PointModel> > LinePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<LineModel, LineModel> > LineLinePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<TriangleModel, SphereModel> > TriangleSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<TriangleModel, PointModel> > TrianglePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<TriangleModel, LineModel> > TriangleLinePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<TriangleModel, TriangleModel> > TriangleTrianglePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<SphereModel, SphereModel> > SphereSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<SphereModel, PointModel> > SpherePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<CapsuleModel, CapsuleModel> > CapsuleCapsulePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<CapsuleModel, TriangleModel> > CapsuleTrianglePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<CapsuleModel, SphereModel> > CapsuleSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<OBBModel, OBBModel> > OBBOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<SphereModel, OBBModel> > SphereOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<CapsuleModel, OBBModel> > CapsuleOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<TriangleModel, OBBModel> > TriangleOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<SphereModel, RigidSphereModel> > SphereRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<LineModel, RigidSphereModel> > LineRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<TriangleModel, RigidSphereModel> > TriangleRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<RigidSphereModel, PointModel> > RigidSpherePointPenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<CapsuleModel, RigidSphereModel> > CapsuleRigidSpherePenaltyCompliantContactClass("PenaltyCompliantContact",true);
sofa::core::collision::ContactCreator< PenaltyCompliantContact<RigidSphereModel, OBBModel> > RigidSphereOBBPenaltyCompliantContactClass("PenaltyCompliantContact",true);


} // namespace collision

} // namespace component

} // namespace sofa
