#ifndef SOFA_COMPONENTS_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_H
#define SOFA_COMPONENTS_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_H

#include "Collision/Contact.h"
#include "Collision/Intersection.h"
#include "BarycentricContactMapper.h"
#include "LagrangianMultiplierContactConstraint.h"
#include "Common/Factory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

template < class TCollisionModel1, class TCollisionModel2 >
class BarycentricLagrangianMultiplierContact : public Collision::Contact, public Abstract::VisualModel
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef Collision::Intersection Intersection;
    typedef Core::MechanicalModel<typename CollisionModel1::DataTypes> MechanicalModel1;
    typedef Core::MechanicalModel<typename CollisionModel2::DataTypes> MechanicalModel2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    BarycentricContactMapper<CollisionModel1> mapper1;
    BarycentricContactMapper<CollisionModel2> mapper2;

    LagrangianMultiplierContactConstraint<Vec3Types>* ff;
    Abstract::BaseContext* parent;
public:
    BarycentricLagrangianMultiplierContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricLagrangianMultiplierContact();

    std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(const std::vector<Collision::DetectionOutput*>& outputs);

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
