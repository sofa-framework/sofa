#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_INTERSECTION_INL
#define SOFA_CORE_COMPONENTMODEL_COLLISION_INTERSECTION_INL

#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
bool FnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return CanIntersectFn(e1, e2);
}

/// Compute the intersection between 2 elements.
template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
DetectionOutput* FnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return IntersectFn(e1, e2);
}

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
bool MirrorFnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return CanIntersectFn(e2, e1);
}

/// Compute the intersection between 2 elements.
template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
DetectionOutput* MirrorFnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
{
    Elem1 e1(elem1);
    Elem2 e2(elem2);
    return IntersectFn(e2, e1);
}

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem1&, Elem2&),
         DetectionOutput* (*IntersectFn)(Elem1&, Elem2&)
         >
std::string FnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::name() const
{
    return gettypename(typeid(Elem1))+std::string("-")+gettypename(typeid(Elem2));
}

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         DetectionOutput* (*IntersectFn)(Elem2&, Elem1&)
         >
std::string MirrorFnElementIntersector<Elem1,Elem2,CanIntersectFn,IntersectFn>
::name() const
{
    return gettypename(typeid(Elem2))+std::string("-")+gettypename(typeid(Elem1));
}

template<class Model1, class Model2,
         bool (*CanIntersectFn)(typename Model1::Element&, typename Model2::Element&),
         DetectionOutput* (*IntersectFn)(typename Model1::Element&, typename Model2::Element&),
         bool mirror
         >
void IntersectorMap::add()
{
    (*this)[std::make_pair(helper::TypeInfo(typeid(Model1)),helper::TypeInfo(typeid(Model2)))] =
        new FnElementIntersector<typename Model1::Element, typename Model2::Element, CanIntersectFn, IntersectFn>;
    if (mirror)
        (*this)[std::make_pair(helper::TypeInfo(typeid(Model2)),helper::TypeInfo(typeid(Model1)))] =
            new MirrorFnElementIntersector<typename Model2::Element, typename Model1::Element, CanIntersectFn, IntersectFn>;
}

template<class Model1, class Model2,
         bool mirror
         >
void IntersectorMap::ignore()
{
    (*this)[std::make_pair(helper::TypeInfo(typeid(Model1)),helper::TypeInfo(typeid(Model2)))] =
        NULL;
    if (mirror)
        (*this)[std::make_pair(helper::TypeInfo(typeid(Model2)),helper::TypeInfo(typeid(Model1)))] =
            NULL;
}

} // namespace collision

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
