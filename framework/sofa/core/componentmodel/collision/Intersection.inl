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

template<class Elem1, class Elem2, class T>
class MemberElementIntersector : public ElementIntersector
{
public:
    typedef typename Elem1::Model Model1;
    typedef typename Elem2::Model Model2;
    MemberElementIntersector(T* ptr) : impl(ptr) {}
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->testIntersection(e1, e2);
    }

    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    /// If the given contacts vector is NULL, then this method should allocate it.
    int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector*& contacts)
    {
        Model1* m1 = static_cast<Model1*>(model1);
        Model2* m2 = static_cast<Model2*>(model2);
        if (contacts == NULL)
        {
            contacts = impl->createOutputVector(m1,m2);
        }
        return impl->beginIntersection(m1, m2, impl->getOutputVector(m1, m2, contacts));
    }

    /// Compute the intersection between 2 elements.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2,  DetectionOutputVector* contacts)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->computeIntersection(e1, e2, impl->getOutputVector(e1.getCollisionModel(), e2.getCollisionModel(), contacts));
    }

    std::string name() const
    {
        return gettypename(typeid(Elem1))+std::string("-")+gettypename(typeid(Elem2));
    }

    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector* contacts)
    {
        Model1* m1 = static_cast<Model1*>(model1);
        Model2* m2 = static_cast<Model2*>(model2);
        return impl->endIntersection(m1, m2, impl->getOutputVector(m1, m2, contacts));
    }

protected:
    T* impl;
};

template<class Elem1, class Elem2, class T>
class MirrorMemberElementIntersector : public ElementIntersector
{
public:
    typedef typename Elem1::Model Model1;
    typedef typename Elem2::Model Model2;
    MirrorMemberElementIntersector(T* ptr) : impl(ptr) {}
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->testIntersection(e2, e1);
    }

    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    /// If the given contacts vector is NULL, then this method should allocate it.
    int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector*& contacts)
    {
        Model1* m1 = static_cast<Model1*>(model1);
        Model2* m2 = static_cast<Model2*>(model2);
        if (contacts == NULL)
        {
            contacts = impl->createOutputVector(m2,m1);
        }
        return impl->beginIntersection(m2, m1, impl->getOutputVector(m2, m1, contacts));
    }

    /// Compute the intersection between 2 elements.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, DetectionOutputVector* contacts)
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->computeIntersection(e2, e1, impl->getOutputVector(e2.getCollisionModel(), e1.getCollisionModel(), contacts));
    }

    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector* contacts)
    {
        Model1* m1 = static_cast<Model1*>(model1);
        Model2* m2 = static_cast<Model2*>(model2);
        return impl->endIntersection(m2, m1, impl->getOutputVector(m2, m1, contacts));
    }

    std::string name() const
    {
        return gettypename(typeid(Elem2))+std::string("-")+gettypename(typeid(Elem1));
    }

protected:
    T* impl;
};

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
