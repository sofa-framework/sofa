#ifndef SOFA_COMPONENTS_CONTACTMANAGERSOFA_H
#define SOFA_COMPONENTS_CONTACTMANAGERSOFA_H

#include "Collision/ContactManager.h"
#include "Sofa/Abstract/VisualModel.h"

#include <vector>

namespace Sofa
{

namespace Components
{

class ContactManagerSofa : public Collision::ContactManager, public Abstract::VisualModel
{
protected:
    std::string contacttype;
    std::map< std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*>, std::vector<Collision::DetectionOutput*> > outputsMap;
    std::map<std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*>,Collision::Contact*> contactMap;
    std::vector<Collision::Contact*> contactVec;

    void clear();
public:
    ContactManagerSofa(const std::string& contacttype);
    ~ContactManagerSofa();

    void createContacts(const std::vector<Collision::DetectionOutput*>& outputs);

    const std::vector<Collision::Contact*>& getContacts() { return contactVec; }

    virtual const char* getTypeName() const { return "CollisionResponse"; }

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
