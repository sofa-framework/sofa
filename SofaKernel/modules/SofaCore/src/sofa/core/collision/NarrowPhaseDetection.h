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
#ifndef SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H
#define SOFA_COMPONENT_COLLISION_NARROWPHASEDETECTION_H

#include <sofa/core/collision/Detection.h>
#include <sofa/core/collision/DetectionOutput.h>
#include <sofa/helper/map_ptr_stable_compare.h>
#include <vector>
#include <map>

namespace sofa::core::collision
{

/**
 * @brief Given a set of potentially colliding pairs of models, compute set of contact points
 */
class SOFA_CORE_API NarrowPhaseDetection : virtual public Detection
{
public:
    SOFA_ABSTRACT_CLASS(NarrowPhaseDetection, Detection);

    typedef sofa::helper::map_ptr_stable_compare< std::pair< core::CollisionModel*, core::CollisionModel* >, DetectionOutputVector* > DetectionOutputMap;

protected:
    /// Destructor
    ~NarrowPhaseDetection() override { }
public:
    /// Clear all the potentially colliding pairs detected in the previous simulation step
    virtual void beginNarrowPhase();

    /// Add a new potentially colliding pairs of models
    virtual void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) = 0;

    /// Add a new list of potentially colliding pairs of models
    virtual void addCollisionPairs(const sofa::helper::vector< std::pair<core::CollisionModel*, core::CollisionModel*> >& v);

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

protected:
    std::map<Instance, DetectionOutputMap> m_storedOutputsMap;

protected:
    DetectionOutputMap m_outputsMap;

    size_t m_primitiveTestCount; // used only for statistics purpose
};

} // namespace sofa

#endif
