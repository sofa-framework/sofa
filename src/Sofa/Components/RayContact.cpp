#include "RayContact.h"
#include "RayModel.h"
#include "SphereModel.h"
#include "TriangleModel.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;

SOFA_DECL_CLASS(RayContact)

Creator<Contact::Factory, RayContact<SphereModel>> RaySphereContactClass("default",true);
Creator<Contact::Factory, RayContact<SphereModel>> RaySphereContactClass2("LagrangianMultiplier",true);

Creator<Contact::Factory, RayContact<TriangleModel>> RayTriangleContactClass("default",true);
Creator<Contact::Factory, RayContact<TriangleModel>> RayTriangleContactClass2("LagrangianMultiplier",true);

BaseRayContact::BaseRayContact(CollisionModel1* model1, Collision::Intersection* /*instersectionMethod*/)
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


} // namespace Components

} // namespace Sofa
