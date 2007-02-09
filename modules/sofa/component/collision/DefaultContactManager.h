#ifndef SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H
#define SOFA_COMPONENT_COLLISION_DEFAULTCONTACTMANAGER_H

#include <sofa/core/componentmodel/collision/ContactManager.h>
#include <sofa/core/VisualModel.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace collision
{

class DefaultContactManager : public core::componentmodel::collision::ContactManager, public core::VisualModel
{
protected:
    std::string contacttype;
    std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, std::vector<core::componentmodel::collision::DetectionOutput*> > outputsMap;
    std::map<std::pair<core::CollisionModel*,core::CollisionModel*>,core::componentmodel::collision::Contact*> contactMap;
    std::vector<core::componentmodel::collision::Contact*> contactVec;

    void clear();
public:
    DefaultContactManager(const std::string& contacttype);
    ~DefaultContactManager();

    void createContacts(const std::vector<core::componentmodel::collision::DetectionOutput*>& outputs);

    const std::vector<core::componentmodel::collision::Contact*>& getContacts() { return contactVec; }

    //virtual const char* getTypeName() const { return "CollisionResponse"; }

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
