#ifndef SOFA_COMPONENTS_CUBEMODEL_H
#define SOFA_COMPONENTS_CUBEMODEL_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Sofa/Abstract/VisualModel.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/Vec3Types.h"
#include "Cube.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

class CubeModel : public Core::MechanicalObject<Vec3Types>, public Abstract::CollisionModel, public Abstract::VisualModel
{
protected:
    std::vector<Abstract::CollisionElement*> elems;
    Abstract::CollisionModel* previous;
    Abstract::CollisionModel* next;
    Abstract::BehaviorModel* object;
    bool static_;
public:

    CubeModel();

    bool isStatic() { return static_; }
    void setStatic(bool val=true) { static_ = val; }

    void clear();
    void addCube(const Vector3& min, const Vector3 &max);
    void setCube(unsigned int index, const Vector3& min, const Vector3 &max);

    std::vector<Abstract::CollisionElement*> & getCollisionElements()
    { return elems; }

//	virtual Abstract::BehaviorModel* getObject()
//	{ return object; }

//	virtual void setObject(Abstract::BehaviorModel* obj)
//	{ object = obj; this->Core::MechanicalObject<Vec3Types>::setObject(obj); }

    Abstract::CollisionModel* getNext()
    { return next; }

    Abstract::CollisionModel* getPrevious()
    { return previous; }

    void setNext(Abstract::CollisionModel* n)
    { next = n; }

    void setPrevious(Abstract::CollisionModel* p)
    { previous = p; }

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
