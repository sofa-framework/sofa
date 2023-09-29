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
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/helper/map_ptr_stable_compare.h>
#include <vector>
#include <map>
#include <unordered_map>

namespace sofa::core::collision
{

/**
 * @brief Given a set of potentially colliding pairs of models, compute set of contact points
 */
class SOFA_CORE_API NarrowPhaseDetection : virtual public Detection
{


public:
    SOFA_ABSTRACT_CLASS(NarrowPhaseDetection, Detection);

    /// A stable sorted associative container of type key-value, where:
    /// * KEY: pair of collision models
    /// * VALUE: collision detection output
    /// The order of the elements in the container is determined by the order the elements are inserted.
    /// Contact response processes collision detection output in the order of this map.
    /// A stable order allows reproducible contact response, therefore a deterministic simulation.
    typedef sofa::helper::map_ptr_stable_compare< std::pair< core::CollisionModel*, core::CollisionModel* >, DetectionOutputVector*> DetectionOutputMap;

protected:
    /// Destructor
    ~NarrowPhaseDetection() override;

public:

    void draw(const core::visual::VisualParams* vparams) override;

    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginNarrowPhase();

    /// Add a new potentially colliding pairs of models
    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) = 0;

    /// Add a new list of potentially colliding pairs of models
    virtual void addCollisionPairs(const sofa::type::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v);

    virtual void endNarrowPhase();

    size_t getPrimitiveTestCount() const;

    const DetectionOutputMap& getDetectionOutputs() const;

    DetectionOutputVector*& getDetectionOutputs(CollisionModel *cm1, CollisionModel *cm2);

    //Returns true if the last narrow phase detected no collision, to use after endNarrowPhase.
    inline bool zeroCollision()const{
        return m_outputsMap.empty();
    }

protected:
    bool _zeroCollision;//true if the last narrow phase detected no collision, to use after endNarrowPhase

    void changeInstanceNP(Instance inst) override;

    std::map<Instance, DetectionOutputMap> m_storedOutputsMap;

    DetectionOutputMap m_outputsMap;

    size_t m_primitiveTestCount; // used only for statistics purpose


};

} // namespace sofa::core::collision
