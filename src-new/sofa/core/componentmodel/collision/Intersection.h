#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_INTERSECTION_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_INTERSECTION_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/componentmodel/collision/DetectionOutput.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;

class ElementIntersector
{
protected:
    virtual ~ElementIntersector() {}
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2) = 0;

    /// Compute the intersection between 2 elements.
    virtual DetectionOutput* intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2) = 0;

    virtual std::string name() const = 0;
};

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
class FnElementIntersector : public ElementIntersector
{
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    /// Compute the intersection between 2 elements.
    DetectionOutput* intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    std::string name() const;
};

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
class MirrorFnElementIntersector : public ElementIntersector
{
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    /// Compute the intersection between 2 elements.
    DetectionOutput* intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    std::string name() const;
};

class IntersectorMap : public std::map< std::pair< helper::TypeInfo, helper::TypeInfo >, ElementIntersector* >
{
public:
    template<class Model1, class Model2,
             bool (*CanIntersectFn)(typename Model1::Element&, typename Model2::Element&),
             DetectionOutput* (*IntersectFn)(typename Model1::Element&, typename Model2::Element&),
             bool mirror
             >
    void add();

    template<class Model1, class Model2,
             bool mirror
             >
    void ignore();

    ElementIntersector* get(core::CollisionModel* model1, core::CollisionModel* model2);
};

class Intersection : public virtual objectmodel::BaseObject
{
public:
    virtual ~Intersection();

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2) = 0;

    /// Test if intersection between 2 types of elements is supported, i.e. an intersection test is implemented for this combinaison of types.
    /// Note that this method is deprecated in favor of findIntersector
    virtual bool isSupported(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present).
    /// Note that this method is deprecated in favor of findIntersector
    virtual bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    /// Compute the intersection between 2 elements.
    /// Note that this method is deprecated in favor of findIntersector
    virtual DetectionOutput* intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2);

    /// returns true if algorithm uses proximity detection
    virtual bool useProximity() const { return false; }

    /// returns true if algorithm uses continous detection
    virtual bool useContinuous() const { return false; }

    /// Return the alarm distance (must return 0 if useProximity() is false)
    virtual double getAlarmDistance() const { return 0.0; }

    /// Return the contact distance (must return 0 if useProximity() is false)
    virtual double getContactDistance() const { return 0.0; }

};

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
