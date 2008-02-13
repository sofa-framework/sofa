#include <sofa/component/collision/BarycentricLagrangianMultiplierContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace collision;
using simulation::tree::GNode;

SOFA_DECL_CLASS(BarycentricLagrangianMultiplierContact)

Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<PointModel, PointModel> > PointPointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineMeshModel, PointModel> > LinePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineMeshModel, LineMeshModel> > LineLineLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleMeshModel, PointModel> > TriangleMeshPointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleMeshModel, LineMeshModel> > TriangleMeshLineLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleMeshModel, TriangleMeshModel> > TriangleMeshTriangleMeshLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereModel, SphereModel> > SphereSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereModel, PointModel> > SpherePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineMeshModel, SphereModel> > LineSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleMeshModel, SphereModel> > TriangleMeshSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereTreeModel,SphereTreeModel> > SphereTreeSphereTreeLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereTreeModel,TriangleMeshModel> > SphereTreeTriangleMeshLagrangianMultiplierContactClass("LagrangianMultiplier",true);

} // namespace collision

} // namespace component

} // namespace sofa

