#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTMANAGER_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_CONTACTMANAGER_H

#include <sofa/core/componentmodel/collision/Contact.h>
#include <sofa/core/objectmodel/BaseObject.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

class ContactManager : public virtual objectmodel::BaseObject
{
public:
    virtual ~ContactManager() { }

    virtual void createContacts(const std::vector<DetectionOutput*>& outputs) = 0;

    virtual const std::vector<Contact*>& getContacts() = 0;

    /// virtual because subclasses might do precomputations based on intersection algorithms
    virtual void setIntersectionMethod(Intersection* v) { intersectionMethod = v;    }
    Intersection* getIntersectionMethod() const         { return intersectionMethod; }

protected:
    /// Current intersection method
    Intersection* intersectionMethod;
};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
