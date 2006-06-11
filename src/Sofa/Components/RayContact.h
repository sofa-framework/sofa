#ifndef SOFA_COMPONENTS_RAYCONTACT_H
#define SOFA_COMPONENTS_RAYCONTACT_H

#include "Collision/Contact.h"
#include "Common/Factory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class RayModel;

class BaseRayContact : public Collision::Contact
{
public:
    typedef RayModel CollisionModel1;

protected:
    CollisionModel1* model1;
    std::vector<Collision::DetectionOutput*> collisions;

public:
    BaseRayContact(CollisionModel1* model1, Collision::Intersection* instersectionMethod);

    ~BaseRayContact();

    void setDetectionOutputs(const std::vector<Collision::DetectionOutput*>& outputs)
    {
        collisions = outputs;
    }

    const std::vector<Collision::DetectionOutput*>& getDetectionOutputs() const { return collisions; }

    void createResponse(Abstract::BaseContext* group)
    {
    }

    void removeResponse()
    {
    }
};

template<class CM2>
class RayContact : public BaseRayContact
{
public:
    typedef RayModel CollisionModel1;
    typedef CM2 CollisionModel2;
    typedef Collision::Intersection Intersection;
protected:
    CollisionModel2* model2;
    Abstract::BaseContext* parent;
public:
    RayContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : BaseRayContact(model1, intersectionMethod), model2(model2)
    {
    }

    std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }
};

} // namespace Components

} // namespace Sofa

#endif
