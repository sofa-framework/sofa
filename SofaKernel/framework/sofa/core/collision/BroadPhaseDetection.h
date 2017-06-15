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
#ifndef SOFA_CORE_COLLISION_BROADPHASEDETECTION_H
#define SOFA_CORE_COLLISION_BROADPHASEDETECTION_H

#include <sofa/core/collision/Detection.h>
#include <vector>
#include <algorithm>

namespace sofa
{

namespace core
{

namespace collision
{
/**
 * @brief given a set of root collision models, computes potentially colliding pairs.
 */
class BroadPhaseDetection : virtual public Detection
{
public:
    SOFA_ABSTRACT_CLASS(BroadPhaseDetection, Detection);
protected:
    /// Destructor
    virtual ~BroadPhaseDetection() { }
public:
    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginBroadPhase()
    {
        cmPairs.clear();
    }

    /// Add a new collision model to the set of root collision models managed by this class
    virtual void addCollisionModel(core::CollisionModel *cm) = 0;

    /// Add a list of collision models to the set of root collision models managed by this class
    virtual void addCollisionModels(const sofa::helper::vector<core::CollisionModel *> v)
    {
        for (sofa::helper::vector<core::CollisionModel *>::const_iterator it = v.begin(); it<v.end(); it++)
            addCollisionModel(*it);
    }

    /// Actions to accomplish when the broadPhase is finished. By default do nothing.
    virtual void endBroadPhase()
    {
    }

    /// Get the potentially colliding pairs detected
    sofa::helper::vector<std::pair<core::CollisionModel*, core::CollisionModel*> >& getCollisionModelPairs() { return cmPairs; }

    /// Returns true because it needs a deep bounding tree i.e. a depth that can be superior to 1.
    inline virtual bool needsDeepBoundingTree()const{return true;}
protected:

    /// Potentially colliding pairs
    sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> > cmPairs;
    std::map<Instance,sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> > > storedCmPairs;

    virtual void changeInstanceBP(Instance inst)
    {
        storedCmPairs[instance].swap(cmPairs);
        cmPairs.swap(storedCmPairs[inst]);
    }
};

} // namespace collision

} // namespace core

} // namespace sofa

#endif
