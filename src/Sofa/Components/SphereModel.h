#ifndef SOFA_COMPONENTS_SPHEREMODEL_H
#define SOFA_COMPONENTS_SPHEREMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"
#include "Sphere.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class SphereModel : public Core::MechanicalObject<Vec3Types>, public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    std::vector<Abstract::CollisionElement*> elems;
    Abstract::CollisionModel* previous;
    Abstract::CollisionModel* next;
    Abstract::BehaviorModel* object;

    class Loader;
    void init(const char* filename);

    VecCoord* internalForces;
    VecCoord* externalForces;
public:

    SphereModel(const char* filename);

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
