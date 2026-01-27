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

#include <sofa/component/collision/detection/algorithm/BaseSubCollisionPipeline.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/core/collision/ContactManager.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>

#include <set>
#include <string>

namespace sofa::component::collision::detection::algorithm
{

/**
 * @brief A self-contained collision pipeline for a specific set of collision models.
 *
 * SubCollisionPipeline implements a complete collision detection and response workflow
 * for an explicitly defined subset of collision models. Unlike the standard CollisionPipeline
 * which processes all collision models in the scene graph, this component only handles
 * the collision models explicitly linked to it.
 *
 * This class is designed to be used as part of a CompositeCollisionPipeline, which can
 * aggregate multiple SubCollisionPipelines to handle different groups of collision models
 * independently (and potentially in parallel).
 *
 * Required components (via links):
 * - At least one CollisionModel
 * - An Intersection method (e.g., MinProximityIntersection, NewProximityIntersection)
 * - A BroadPhaseDetection (e.g., BruteForceBroadPhase, BVHNarrowPhase)
 * - A NarrowPhaseDetection (e.g., BVHNarrowPhase, DirectSAP)
 * - A ContactManager (e.g., DefaultContactManager)
 *
 * @see CompositeCollisionPipeline
 * @see BaseSubCollisionPipeline
 */
class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API SubCollisionPipeline : public BaseSubCollisionPipeline
{
public:
    using Inherited = BaseSubCollisionPipeline;
    SOFA_CLASS(SubCollisionPipeline, Inherited);
protected:
    SubCollisionPipeline();
public:
    virtual ~SubCollisionPipeline() override = default;

    /// @brief Validates that all required components are linked and sets the component state accordingly.
    void doInit() override;

    /// @brief Event handling (currently no-op for this pipeline).
    void doHandleEvent(sofa::core::objectmodel::Event*) override {}

    /// @brief Clears contact responses from the previous time step.
    void computeCollisionReset() override;

    /// @brief Performs collision detection: computes bounding trees, runs broad and narrow phase detection.
    void computeCollisionDetection() override;

    /// @brief Creates contact responses based on detected collisions.
    void computeCollisionResponse() override;

    /// @brief Returns the list of collision models handled by this pipeline.
    std::vector<sofa::core::CollisionModel*> getCollisionModels() override;

    /// Maximum depth of bounding trees used in collision detection.
    sofa::Data<unsigned int>  d_depth;

    /// List of collision models to process in this pipeline.
    sofa::MultiLink < SubCollisionPipeline, sofa::core::CollisionModel, sofa::BaseLink::FLAG_DUPLICATE > l_collisionModels;

    /// Intersection method defining how to detect intersections between geometric primitives.
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::Intersection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_intersectionMethod;

    /// Contact manager responsible for creating and managing contact objects.
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::ContactManager, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_contactManager;

    /// Broad phase detection algorithm for quickly identifying potentially colliding pairs.
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::BroadPhaseDetection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_broadPhaseDetection;

    /// Narrow phase detection algorithm for precise intersection testing.
    sofa::SingleLink< SubCollisionPipeline, sofa::core::collision::NarrowPhaseDetection, sofa::BaseLink::FLAG_STOREPATH | sofa::BaseLink::FLAG_STRONGLINK > l_narrowPhaseDetection;

    /// Default value for the bounding tree depth parameter.
    static inline constexpr unsigned int s_defaultDepthValue = 6;
};

} // namespace sofa::component::collision::detection::algorithm
