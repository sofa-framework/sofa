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
#include <sofa/component/collision/detection/algorithm/config.h>

#include <sofa/core/collision/Intersection.h>

namespace sofa::component::collision::detection::algorithm
{

class MirrorIntersector : public core::collision::ElementIntersector
{
public:
    core::collision::ElementIntersector* intersector{ nullptr };

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, const core::collision::Intersection* currentIntersection) override
    {
        assert(intersector != nullptr);
        return intersector->canIntersect(elem2, elem1, currentIntersection);
    }

    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    /// If the given contacts vector is nullptr, then this method should allocate it.
    int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::DetectionOutputVector*& contacts) override
    {
        assert(intersector != nullptr);
        return intersector->beginIntersect(model2, model1, contacts);
    }

    /// Compute the intersection between 2 elements. Return the number of contacts written in the contacts vector.
    int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, core::collision::DetectionOutputVector* contacts, const core::collision::Intersection* currentIntersection) override
    {
        assert(intersector != nullptr);
        return intersector->intersect(elem2, elem1, contacts, currentIntersection);
    }

    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
    int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::DetectionOutputVector* contacts) override
    {
        assert(intersector != nullptr);
        return intersector->endIntersect(model2, model1, contacts);
    }

    std::string name() const override
    {
        assert(intersector != nullptr);
        return intersector->name() + std::string("<SWAP>");
    }
};

} // namespace sofa::component::collision::detection::algorithm
