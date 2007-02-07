#ifndef SOFA_COMPONENTS_COLLISION_CONTACTMANAGER_H
#define SOFA_COMPONENTS_COLLISION_CONTACTMANAGER_H

#include "Contact.h"
#include "Sofa-old/Abstract/BaseObject.h"

#include <vector>

namespace Sofa
{

namespace Components
{

namespace Collision
{

class ContactManager : public virtual Abstract::BaseObject
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

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
