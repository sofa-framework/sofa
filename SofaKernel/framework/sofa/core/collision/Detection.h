/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_COLLISION_DETECTION_H
#define SOFA_CORE_COLLISION_DETECTION_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/CollisionAlgorithm.h>
#include <sofa/core/collision/Intersection.h>
#include <vector>
#include <map>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace collision
{

class Detection : public virtual CollisionAlgorithm
{
public:
    SOFA_CLASS(Detection, CollisionAlgorithm);
    SOFA_BASE_CAST_IMPLEMENTATION(Detection)

protected:
    /// Current intersection method
    Intersection* intersectionMethod;
    /// All intersection methods
    std::map<Instance,Intersection*> storedIntersectionMethod;


    Detection()
        : intersectionMethod(NULL)
    {
    }
public:
    /// virtual because subclasses might do precomputations based on intersection algorithms
    virtual void setIntersectionMethod(Intersection* v) { intersectionMethod = v;    }
    Intersection* getIntersectionMethod() const         { return intersectionMethod; }

protected:
    virtual void changeInstanceBP(Instance) {}
    virtual void changeInstanceNP(Instance) {}
    virtual void changeInstance(Instance inst)
    {
        storedIntersectionMethod[instance] = intersectionMethod;
        intersectionMethod = storedIntersectionMethod[inst];
        // callback overriden by BroadPhaseDetection
        changeInstanceBP(inst);
        // callback overriden by NarrowPhaseDetection
        changeInstanceNP(inst);
    }
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
