#ifndef SOFA_COMPONENTS_RAYMODEL_H
#define SOFA_COMPONENTS_RAYMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"
#include "Ray.h"
#include <set>

namespace Sofa
{

namespace Components
{

using namespace Common;

class RayContact;

class RayModel : public Core::MechanicalObject<Vec3Types>, public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    std::vector<Abstract::CollisionElement*> elems;
    Abstract::CollisionModel* previous;
    Abstract::CollisionModel* next;
    Abstract::BehaviorModel* object;
    std::set<RayContact*> contacts;
    VecCoord* internalForces;
    VecCoord* externalForces;
public:

    RayModel();

    virtual void init()
    {
        Core::MechanicalObject<Vec3Types>::init();
    }

    void clear() { resize(0); }

    void resize(int size);

    void addRay(Vector3 origin, Vector3 direction, double length);

    int getNbRay() const { return elems.size(); }

    void setNbRay(int n) { resize(2*n); }

    Ray* getRay(int index) { return static_cast<Ray*>(elems[index]); }

    virtual void addContact(RayContact* contact) { contacts.insert(contact); }

    virtual void removeContact(RayContact* contact) { contacts.erase(contact); }

    // -- MechanicalModel interface

    virtual void setObject(Abstract::BehaviorModel* obj);

    virtual void beginIteration(double dt);

    virtual void endIteration(double dt);

    virtual void accumulateForce();

    // -- CollisionModel interface

    virtual Abstract::BehaviorModel* getObject()
    { return object; }

    void computeBoundingBox();

    std::vector<Abstract::CollisionElement*> & getCollisionElements()
    { return elems; }

    Abstract::CollisionModel* getNext()
    { return next; }

    Abstract::CollisionModel* getPrevious()
    { return previous; }

    void setNext(Abstract::CollisionModel* n)
    { next = n; }

    void setPrevious(Abstract::CollisionModel* p)
    { previous = p; }

    void applyTranslation(double dx, double dy, double dz);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
