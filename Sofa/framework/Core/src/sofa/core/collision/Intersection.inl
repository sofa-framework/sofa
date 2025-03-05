/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/Factory.h>

#include <sofa/version.h>

namespace sofa::core::collision
{

template<class Elem1, class Elem2, class T>
class MemberElementIntersector : public ElementIntersector
{
public:
    typedef typename Elem1::Model Model1;
    typedef typename Elem2::Model Model2;
    MemberElementIntersector(T* ptr) : impl(ptr) {}
    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    SOFA_ATTRIBUTE_DISABLED__CORE_INTERSECTION_AS_PARAMETER()
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2) = delete;
    
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, const core::collision::Intersection* currentIntersection) override
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->testIntersection(e1, e2, currentIntersection);
    }

    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    /// If the given contacts vector is nullptr, then this method should allocate it.
    int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector*& contacts) override
    {
        Model1* m1 = static_cast<Model1*>(model1);
        Model2* m2 = static_cast<Model2*>(model2);
        if (contacts == nullptr)
        {
            contacts = impl->createOutputVector(m1,m2);
        }
        return impl->beginIntersection(m1, m2, impl->getOutputVector(m1, m2, contacts));
    }

    /// Compute the intersection between 2 elements.
    SOFA_ATTRIBUTE_DISABLED__CORE_INTERSECTION_AS_PARAMETER()
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2,  DetectionOutputVector* contacts) = delete;
    
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2,  DetectionOutputVector* contacts, const core::collision::Intersection* currentIntersection) override
    {
        Elem1 e1(elem1);
        Elem2 e2(elem2);
        return impl->computeIntersection(e1, e2, impl->getOutputVector(e1.getCollisionModel(), e2.getCollisionModel(), contacts), currentIntersection);
    }

    std::string name() const override
    {
        return sofa::helper::gettypename(typeid(Elem1))+std::string("-")+sofa::helper::gettypename(typeid(Elem2));
    }

    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, DetectionOutputVector* contacts) override
    {
        Model1* m1 = static_cast<Model1*>(model1);
        Model2* m2 = static_cast<Model2*>(model2);
        return impl->endIntersection(m1, m2, impl->getOutputVector(m1, m2, contacts));
    }

protected:
    T* impl;
};

template<class Model1, class Model2, class T>
void IntersectorMap::add(T* ptr)
{
    add_impl<Model1, Model2>(new MemberElementIntersector<typename Model1::Element, typename Model2::Element, T>(ptr));
}

template<class Model1, class Model2>
void IntersectorMap::ignore()
{
    add_impl<Model1, Model2>(0);
}

template<class Model1, class Model2>
void IntersectorMap::add_impl(ElementIntersector* intersector)
{
    add_impl(classid(Model1), classid(Model2), intersector);
}
} // namespace sofa
