#ifndef SOFA_COMPONENT_COLLISION_RAYCONTACT_H
#define SOFA_COMPONENT_COLLISION_RAYCONTACT_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class RayModel;

class BaseRayContact : public core::componentmodel::collision::Contact
{
public:
    typedef RayModel CollisionModel1;

protected:
    CollisionModel1* model1;
    std::vector<core::componentmodel::collision::DetectionOutput*> collisions;

public:
    BaseRayContact(CollisionModel1* model1, core::componentmodel::collision::Intersection* instersectionMethod);

    ~BaseRayContact();

    void setDetectionOutputs(const std::vector<core::componentmodel::collision::DetectionOutput*>& outputs)
    {
        collisions = outputs;
    }

    const std::vector<core::componentmodel::collision::DetectionOutput*>& getDetectionOutputs() const { return collisions; }

    void createResponse(core::objectmodel::BaseContext* /*group*/)
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
    typedef core::componentmodel::collision::Intersection Intersection;
protected:
    CollisionModel2* model2;
    core::objectmodel::BaseContext* parent;
public:
    RayContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : BaseRayContact(model1, intersectionMethod), model2(model2)
    {
    }

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
