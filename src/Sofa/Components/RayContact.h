#ifndef SOFA_COMPONENTS_RAYCONTACT_H
#define SOFA_COMPONENTS_RAYCONTACT_H

#include "Collision/Contact.h"
#include "SphereModel.h"
#include "RayModel.h"
#include "Common/Factory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class RayContact : public Collision::Contact, public Abstract::VisualModel
{
public:
    typedef RayModel CollisionModel1;
    typedef SphereModel CollisionModel2;
    std::vector<Collision::DetectionOutput*> collisions;
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Abstract::BaseContext* parent;
public:
    RayContact(CollisionModel1* model1, CollisionModel2* model2);
    ~RayContact();

    std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(const std::vector<Collision::DetectionOutput*>& outputs);

    const std::vector<Collision::DetectionOutput*>& getDetectionOutputs() const { return collisions; };

    void createResponse(Abstract::BaseContext* group);

    void removeResponse();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
