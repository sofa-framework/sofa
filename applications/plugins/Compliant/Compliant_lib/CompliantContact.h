#ifndef SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H
#define SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_H

#include <sofa/component/collision/FrictionContact.h>
#include "initCompliant.h"

namespace sofa
{

namespace component
{

namespace collision
{

template <class TCollisionModel1, class TCollisionModel2>
class SOFA_Compliant_API CompliantContact : public FrictionContact<TCollisionModel1, TCollisionModel2>
{

public:
    typedef FrictionContact<TCollisionModel1, TCollisionModel2> base;
    SOFA_CLASS(SOFA_TEMPLATE2(CompliantContact, TCollisionModel1, TCollisionModel2), core::collision::Contact );

    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef core::collision::DetectionOutputVector OutputVector;
    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;

public:

    CompliantContact() {}
    CompliantContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
        : base(model1, model2, intersectionMethod) { }
    ~CompliantContact();

    void createResponse(core::objectmodel::BaseContext* group);
    void removeResponse();

};
}
}
}

#endif
