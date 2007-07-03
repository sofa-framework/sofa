/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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
         int (*IntersectFn)(Elem1&, Elem2&, std::vector<DetectionOutput>&)
         >
class FnElementIntersector : public ElementIntersector
{
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return CanIntersectFn(e1, e2);
    }

    /// Compute the intersection between 2 elements.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, std::vector<DetectionOutput>& contacts)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return IntersectFn(e1, e2, contacts);
    }

    std::string name() const
    {
        return gettypename(typeid(Elem1))+std::string("-")+gettypename(typeid(Elem2));
    }
};

template<class Elem1, class Elem2,
         bool (*CanIntersectFn)(Elem2&, Elem1&),
         int (*IntersectFn)(Elem1&, Elem2&, std::vector<DetectionOutput>&)
         >
class MirrorFnElementIntersector : public ElementIntersector
{
public:
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return CanIntersectFn(e2, e1);
    }

    /// Compute the intersection between 2 elements.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, std::vector<DetectionOutput>& contacts)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return IntersectFn(e2, e1, contacts);
    }

    std::string name() const
    {
        return gettypename(typeid(Elem2))+std::string("-")+gettypename(typeid(Elem1));
    }
};

template<class Elem1, class Elem2, class T>
class MemberElementIntersector : public ElementIntersector
{
public:
    MemberElementIntersector(T* ptr) : impl(ptr) {}
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->testIntersection(e1, e2);
    }

    /// Compute the intersection between 2 elements.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2,  std::vector<DetectionOutput>& contacts)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->computeIntersection(e1, e2, contacts);
    }

    std::string name() const
    {
        return gettypename(typeid(Elem1))+std::string("-")+gettypename(typeid(Elem2));
    }

protected:
    T* impl;
};

template<class Elem1, class Elem2, class T>
class MirrorMemberElementIntersector : public ElementIntersector
{
public:
    MirrorMemberElementIntersector(T* ptr) : impl(ptr) {}
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->testIntersection(e2, e1);
    }

    /// Compute the intersection between 2 elements.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, std::vector<DetectionOutput>& contacts)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->computeIntersection(e2, e1, contacts);
    }

    std::string name() const
    {
        return gettypename(typeid(Elem2))+std::string("-")+gettypename(typeid(Elem1));
    }

protected:
    T* impl;
};

template<class Model1, class Model2,
         bool (*CanIntersectFn)(typename Model1::Element&, typename Model2::Element&),
         int (*IntersectFn)(typename Model1::Element&, typename Model2::Element&, std::vector<DetectionOutput>&),
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

template<class Model1, class Model2, class T,
         bool mirror
         >
void IntersectorMap::add(T* ptr)
{
    (*this)[std::make_pair(helper::TypeInfo(typeid(Model1)),helper::TypeInfo(typeid(Model2)))] =
        new MemberElementIntersector<typename Model1::Element, typename Model2::Element, T>(ptr);
    if (mirror)
        (*this)[std::make_pair(helper::TypeInfo(typeid(Model2)),helper::TypeInfo(typeid(Model1)))] =
            new MirrorMemberElementIntersector<typename Model2::Element, typename Model1::Element, T>(ptr);
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
