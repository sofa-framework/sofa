#ifndef SOFA_COMPONENTS_COLLISION_INTERSECTION_H
#define SOFA_COMPONENTS_COLLISION_INTERSECTION_H

#include "Sofa-old/Abstract/CollisionModel.h"
#include "DetectionOutput.h"
#include "Sofa-old/Components/Common/FnDispatcher.h"

namespace Sofa
{

namespace Components
{

namespace Collision
{

using namespace Common;

class ElementIntersector
{
protected:
    virtual ~ElementIntersector() {}
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2) = 0;

    /// Compute the intersection between 2 elements.
    virtual Collision::DetectionOutput* intersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2) = 0;

    virtual std::string name() const = 0;
};

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         Collision::DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
class FnElementIntersector : public ElementIntersector
{
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    /// Compute the intersection between 2 elements.
    Collision::DetectionOutput* intersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    std::string name() const;
};

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         Collision::DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
class MirrorFnElementIntersector : public ElementIntersector
{
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    /// Compute the intersection between 2 elements.
    Collision::DetectionOutput* intersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    std::string name() const;
};

class IntersectorMap : public std::map< std::pair< TypeInfo, TypeInfo >, ElementIntersector* >
{
public:
    template<class Model1, class Model2,
             bool (*CanIntersectFn)(typename Model1::Element&, typename Model2::Element&),
             Collision::DetectionOutput* (*IntersectFn)(typename Model1::Element&, typename Model2::Element&),
             bool mirror
             >
    void add();

    template<class Model1, class Model2,
             bool mirror
             >
    void ignore();

    ElementIntersector* get(Abstract::CollisionModel* model1, Abstract::CollisionModel* model2);
};

class Intersection : public virtual Abstract::BaseObject
{
public:
    virtual ~Intersection();

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual ElementIntersector* findIntersector(Abstract::CollisionModel* object1, Abstract::CollisionModel* object2) = 0;

    /// Test if intersection between 2 types of elements is supported, i.e. an intersection test is implemented for this combinaison of types.
    /// Note that this method is deprecated in favor of findIntersector
    virtual bool isSupported(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present).
    /// Note that this method is deprecated in favor of findIntersector
    virtual bool canIntersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    /// Compute the intersection between 2 elements.
    /// Note that this method is deprecated in favor of findIntersector
    virtual Collision::DetectionOutput* intersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2);

    /// returns true if algorithm uses proximity detection
    virtual bool useProximity() const { return false; }

    /// returns true if algorithm uses continous detection
    virtual bool useContinuous() const { return false; }

    /// Return the alarm distance (must return 0 if useProximity() is false)
    virtual double getAlarmDistance() const { return 0.0; }

    /// Return the contact distance (must return 0 if useProximity() is false)
    virtual double getContactDistance() const { return 0.0; }

};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
