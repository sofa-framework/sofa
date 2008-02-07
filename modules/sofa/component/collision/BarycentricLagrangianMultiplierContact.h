#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_H
#define SOFA_COMPONENT_COLLISION_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/constraint/LagrangianMultiplierContactConstraint.h>
#include <sofa/helper/Factory.h>



namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template < class TCollisionModel1, class TCollisionModel2 >
class BarycentricLagrangianMultiplierContact : public core::componentmodel::collision::Contact
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::componentmodel::collision::Intersection Intersection;
    typedef core::componentmodel::collision::DetectionOutputVector OutputVector;
    typedef core::componentmodel::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    constraint::LagrangianMultiplierContactConstraint<Vec3Types>* ff;
    core::objectmodel::BaseContext* parent;
public:
    BarycentricLagrangianMultiplierContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    ~BarycentricLagrangianMultiplierContact();

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    void removeResponse();

    void draw();
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
