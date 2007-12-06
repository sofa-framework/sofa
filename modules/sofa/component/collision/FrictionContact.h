#ifndef SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_H
#define SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/constraint/UnilateralInteractionConstraint.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/collision/BarycentricContactMapper.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class Identifier
{
public:
    Identifier()
    {
        if (!availableId.empty())
        {
            id = availableId.front();
            availableId.pop_front();
        }
        else
            id = cpt++;

        //	std::cout << id << std::endl;
    }

    virtual ~Identifier()
    {
        availableId.push_back(id);
    }

protected:
    static unsigned int cpt;
    unsigned int id;
    static std::list<unsigned int> availableId;
};


template <class TCollisionModel1, class TCollisionModel2>
class FrictionContact : public core::componentmodel::collision::Contact, public Identifier
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::componentmodel::collision::Intersection Intersection;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef core::componentmodel::collision::DetectionOutputVector OutputVector;
    typedef core::componentmodel::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;

protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;

    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    constraint::UnilateralInteractionConstraint<Vec3Types>* c;
    core::objectmodel::BaseContext* parent;

    double mu;

public:

    FrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    virtual ~FrictionContact();

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    void removeResponse();
};

long cantorPolynomia(unsigned long x, unsigned long y)
{
    // Polynome de Cantor de N² sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
    return ((x+y)*(x+y)+3*x+y)/2;
}

} // collision

} // component

} // sofa

#endif // SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_H
