#include <sofa/component/collision/RayContact.h>
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/SphereTreeModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(RayContact)

Creator<core::componentmodel::collision::Contact::Factory, RayContact<SphereModel> > RaySphereContactClass("default",true);
Creator<core::componentmodel::collision::Contact::Factory, RayContact<SphereModel> > RaySphereContactClass2("LagrangianMultiplier",true);

Creator<core::componentmodel::collision::Contact::Factory, RayContact<TriangleModel> > RayTriangleContactClass("default",true);
Creator<core::componentmodel::collision::Contact::Factory, RayContact<TriangleModel> > RayTriangleContactClass2("LagrangianMultiplier",true);

Creator<core::componentmodel::collision::Contact::Factory, RayContact<SphereTreeModel> > RaySphereTreeContactClass("default",true);
Creator<core::componentmodel::collision::Contact::Factory, RayContact<SphereTreeModel> > RaySphereTreeContactClass2("LagrangianMultiplier",true);

BaseRayContact::BaseRayContact(CollisionModel1* model1, core::componentmodel::collision::Intersection* /*instersectionMethod*/)
    : model1(model1)
{
    if (model1!=NULL)
        model1->addContact(this);
}

BaseRayContact::~BaseRayContact()
{
    if (model1!=NULL)
        model1->removeContact(this);
}


} // namespace collision

} // namespace component

} // namespace sofa

