#include "RayContact.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;

SOFA_DECL_CLASS(RayContact)

Creator<Contact::Factory, RayContact> RayContactClass("default",true);

RayContact::RayContact(CollisionModel1* model1, CollisionModel2* model2)
    : model1(model1), model2(model2), parent(NULL)
{
    if (model1!=NULL)
    {
        model1->addContact(this);
    }
}

RayContact::~RayContact()
{
    if (model1!=NULL)
    {
        model1->removeContact(this);
    }
}

void RayContact::setDetectionOutputs(const std::vector<DetectionOutput*>& outputs)
{
    collisions = outputs;
}

void RayContact::createResponse(Core::Group* /*group*/)
{
}

void RayContact::removeResponse()
{
}

void RayContact::draw()
{
//	if (dynamic_cast<Abstract::VisualModel*>(ff)!=NULL)
//		dynamic_cast<Abstract::VisualModel*>(ff)->draw();
}

} // namespace Components

} // namespace Sofa
