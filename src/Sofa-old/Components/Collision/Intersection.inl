#ifndef SOFA_COMPONENTS_COLLISION_INTERSECTION_INL
#define SOFA_COMPONENTS_COLLISION_INTERSECTION_INL

#include "Intersection.h"
#include "Common/Factory.h"

namespace Sofa
{

namespace Components
{

namespace Collision
{

using namespace Common;

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         Collision::DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
bool FnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::canIntersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return CanIntersectFn(e1, e2);
}

/// Compute the intersection between 2 elements.
template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         Collision::DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
Collision::DetectionOutput* FnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::intersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return IntersectFn(e1, e2);
}

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         Collision::DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
bool MirrorFnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::canIntersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return CanIntersectFn(e2, e1);
}

/// Compute the intersection between 2 elements.
template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         Collision::DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
Collision::DetectionOutput* MirrorFnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::intersect(Abstract::CollisionElementIterator elem1, Abstract::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return IntersectFn(e2, e1);
}

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         Collision::DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
std::string FnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::name() const
{
    return gettypename(typeid(Elem1))+std::string("-")+gettypename(typeid(Elem2));
}

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         Collision::DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
std::string MirrorFnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::name() const
{
    return gettypename(typeid(Elem2))+std::string("-")+gettypename(typeid(Elem1));
}

template<class Model1, class Model2,
         bool (*CanIntersectFn)(typename Model1::Element&, typename Model2::Element&),
         Collision::DetectionOutput* (*IntersectFn)(typename Model1::Element&, typename Model2::Element&),
         bool mirror
         >
void IntersectorMap::add()
{
    (*this)[std::make_pair(TypeInfo(typeid(Model1)),TypeInfo(typeid(Model2)))] =
        new FnElementIntersector<typename Model1::Element, typename Model2::Element, CanIntersectFn, IntersectFn>;
    if (mirror)
        (*this)[std::make_pair(TypeInfo(typeid(Model2)),TypeInfo(typeid(Model1)))] =
            new MirrorFnElementIntersector<typename Model2::Element, typename Model1::Element, CanIntersectFn, IntersectFn>;
}

template<class Model1, class Model2,
         bool mirror
         >
void IntersectorMap::ignore()
{
    (*this)[std::make_pair(TypeInfo(typeid(Model1)),TypeInfo(typeid(Model2)))] =
        NULL;
    if (mirror)
        (*this)[std::make_pair(TypeInfo(typeid(Model2)),TypeInfo(typeid(Model1)))] =
            NULL;
}

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
