#ifndef SOFA_COMPONENTS_COLLISION_CONTACT_H
#define SOFA_COMPONENTS_COLLISION_CONTACT_H

#include "DetectionOutput.h"
#include "Sofa/Abstract/BaseContext.h"
#include "../Common/Factory.h"

#include <vector>

namespace Sofa
{

namespace Components
{

namespace Collision
{

using namespace Common;

class Contact
{
public:
    virtual ~Contact() { }

    virtual std::pair< Abstract::CollisionModel*, Abstract::CollisionModel* > getCollisionModels() = 0;

    virtual void setDetectionOutputs(const std::vector<DetectionOutput*>& outputs) = 0;

    virtual void createResponse(Abstract::BaseContext* group) = 0;

    virtual void removeResponse() = 0;

    typedef Factory< std::string, Contact, std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*> > Factory;

    static Contact* Create(const std::string& type, Abstract::CollisionModel* model1, Abstract::CollisionModel* model2);
};

template<class RealContact>
void create(RealContact*& obj, std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*> arg)
{
    typedef typename RealContact::CollisionModel1 RealCollisionModel1;
    typedef typename RealContact::CollisionModel2 RealCollisionModel2;
    RealCollisionModel1* model1 = dynamic_cast<RealCollisionModel1*>(arg.first);
    RealCollisionModel2* model2 = dynamic_cast<RealCollisionModel2*>(arg.second);
    if (model1==NULL || model2==NULL)
    {
        // Try the other way around
        model1 = dynamic_cast<RealCollisionModel1*>(arg.second);
        model2 = dynamic_cast<RealCollisionModel2*>(arg.first);
    }
    if (model1==NULL || model2==NULL) return;
    obj = new RealContact(model1, model2);
}

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
