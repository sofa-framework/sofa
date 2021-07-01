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

#include <sofa/core/collision/Detection.h>
#include <sofa/type/vector.h>

namespace sofa::core::collision
{

/**
 * @brief given a set of root collision models, computes potentially colliding pairs.
 */
class SOFA_CORE_API BroadPhaseDetection : virtual public Detection
{
public:
    SOFA_ABSTRACT_CLASS(BroadPhaseDetection, Detection);

    using CollisionModelPair = std::pair<core::CollisionModel*, core::CollisionModel*>;
protected:
    /// Destructor
    ~BroadPhaseDetection() override = default;
public:
    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginBroadPhase();

    /// Add a new collision model to the set of root collision models managed by this class
    virtual void addCollisionModel(core::CollisionModel *cm) = 0;

    /// Add a list of collision models to the set of root collision models managed by this class
    virtual void addCollisionModels(const sofa::type::vector<core::CollisionModel *>& v);

    /// Actions to accomplish when the broadPhase is finished. By default do nothing.
    virtual void endBroadPhase();

    /// Get the potentially colliding pairs detected
    sofa::type::vector<CollisionModelPair>& getCollisionModelPairs();
    const sofa::type::vector<CollisionModelPair>& getCollisionModelPairs() const;

protected:

    /// Potentially colliding pairs
    sofa::type::vector< CollisionModelPair > cmPairs;
    std::map<Instance,sofa::type::vector< CollisionModelPair > > storedCmPairs;

    void changeInstanceBP(Instance inst) override;
};

} // namespace sofa::core::collision