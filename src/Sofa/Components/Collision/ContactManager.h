#ifndef SOFA_COMPONENTS_COLLISION_CONTACTMANAGER_H
#define SOFA_COMPONENTS_COLLISION_CONTACTMANAGER_H

#include "Contact.h"
#include "Sofa/Abstract/BaseObject.h"

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

    virtual const char* getName() = 0;

    virtual void createContacts(const std::vector<DetectionOutput*>& outputs) = 0;

    virtual const std::vector<Contact*>& getContacts() = 0;
};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
