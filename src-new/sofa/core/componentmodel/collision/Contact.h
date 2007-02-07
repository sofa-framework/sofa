#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACT_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACT_H

#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/helper/Factory.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

class Contact
{
public:
    virtual ~Contact() { }

    virtual std::pair< core::CollisionModel*, core::CollisionModel* > getCollisionModels() = 0;

    virtual void setDetectionOutputs(const std::vector<DetectionOutput*>& outputs) = 0;

    virtual void createResponse(objectmodel::BaseContext* group) = 0;

    virtual void removeResponse() = 0;

    typedef helper::Factory< std::string, Contact, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> > Factory;

    static Contact* Create(const std::string& type, core::CollisionModel* model1, core::CollisionModel* model2, Intersection* intersectionMethod);
};

template<class RealContact>
void create(RealContact*& obj, std::pair<std::pair<core::CollisionModel*,core::CollisionModel*>,Intersection*> arg)
{
    typedef typename RealContact::CollisionModel1 RealCollisionModel1;
    typedef typename RealContact::CollisionModel2 RealCollisionModel2;
    typedef typename RealContact::Intersection RealIntersection;
    RealCollisionModel1* model1 = dynamic_cast<RealCollisionModel1*>(arg.first.first);
    RealCollisionModel2* model2 = dynamic_cast<RealCollisionModel2*>(arg.first.second);
    RealIntersection* inter  = dynamic_cast<RealIntersection*>(arg.second);
    if (model1==NULL || model2==NULL)
    {
        // Try the other way around
        model1 = dynamic_cast<RealCollisionModel1*>(arg.first.second);
        model2 = dynamic_cast<RealCollisionModel2*>(arg.first.first);
    }
    if (model1==NULL || model2==NULL || inter==NULL) return;
    obj = new RealContact(model1, model2, inter);
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
