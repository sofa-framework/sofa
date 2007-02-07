#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/forcefield/PenalityContactForceField.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/collision/BarycentricContactMapper.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template < class TCollisionModel1, class TCollisionModel2 >
class BarycentricPenalityContact : public core::componentmodel::collision::Contact, public core::VisualModel
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::componentmodel::collision::Intersection Intersection;
    typedef core::componentmodel::behavior::MechanicalState<typename CollisionModel1::DataTypes> MechanicalState1;
    typedef core::componentmodel::behavior::MechanicalState<typename CollisionModel2::DataTypes> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    BarycentricContactMapper<CollisionModel1> mapper1;
    BarycentricContactMapper<CollisionModel2> mapper2;

    forcefield::PenalityContactForceField<Vec3Types>* ff;
    core::objectmodel::BaseContext* parent;
public:
    BarycentricPenalityContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricPenalityContact();

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(const std::vector<core::componentmodel::collision::DetectionOutput*>& outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    void removeResponse();

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
