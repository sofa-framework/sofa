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

    class Loader;
    void init(const char* filename);

    VecDeriv* internalForces;
    VecDeriv* externalForces;
    bool static_;
public:

    SphereModel(const char* filename);

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    // -- MechanicalModel interface

    virtual void beginIntegration(double dt);

    virtual void endIntegration(double dt);

    virtual void accumulateForce();

    // -- CollisionModel interface

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
