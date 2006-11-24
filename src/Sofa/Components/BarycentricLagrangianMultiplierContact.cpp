#include "BarycentricLagrangianMultiplierContact.inl"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;
using Graph::GNode;

SOFA_DECL_CLASS(BarycentricLagrangianMultiplierContact)

Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<PointModel, PointModel> > PointPointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineModel, PointModel> > LinePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineModel, LineModel> > LineLineLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, PointModel> > TrianglePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, LineModel> > TriangleLineLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, TriangleModel> > TriangleTriangleLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereModel, SphereModel> > SphereSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereModel, PointModel> > SpherePointLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<LineModel, SphereModel> > LineSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<TriangleModel, SphereModel> > TriangleSphereLagrangianMultiplierContactClass("LagrangianMultiplier",true);
Creator<Contact::Factory, BarycentricLagrangianMultiplierContact<SphereTreeModel,SphereTreeModel> > SphereTreeSphereTreeLagrangianMultiplierContactClass("LagrangianMultiplier",true);
} // namespace Components

} // namespace Sofa
