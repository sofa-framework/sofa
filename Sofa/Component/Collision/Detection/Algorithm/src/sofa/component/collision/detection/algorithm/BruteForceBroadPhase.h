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
#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/component/collision/geometry/CubeCollisionModel.h>

namespace sofa::component::collision::detection::algorithm
{

/**
 * @brief Perform an extensive pair-wise collision test based on the bounding volume of collision models
 *
 * This component is a broad phase algorithm used during collision detection to limit the number of pairs of objects
 * that need to be checked for intersection. The algorithm output is a list of pairs of objects that can potentially
 * be in intersection. This list is then used as an input for a narrow phase algorithm.
 * In this algorithm, all possible pairs of objects are tested (brute force test). If there are n objects, there will be
 * n^2/2 tests. The tests are based on the bounding volume of the objects, usually an axis-aligned bounding box.
 */
class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API BruteForceBroadPhase : public core::collision::BroadPhaseDetection
{
public:
    SOFA_CLASS(BruteForceBroadPhase, core::collision::BroadPhaseDetection);

protected:
    BruteForceBroadPhase();

    ~BruteForceBroadPhase() override = default;

private:
    Data<type::fixed_array<sofa::type::Vec3, 2> > d_box; ///< if not empty, objects that do not intersect this bounding-box will be ignored


public:
    void init() override;
    void reinit() override;

    void beginBroadPhase() override;

    /** \brief In the broad phase, ignores collision with the provided collision model if possible and add pairs of
     * collision models if in intersection.
     *
     * Ignore the collision with the provided collision model if it does not intersect with the box defined in
     * the Data box when it is defined.
     * Add the provided collision model to be investigated in the narrow phase in case of self collision.
     * Check intersection with already added collision models. If it can intersect another collision model, the pair
     * is added to be further investigated in the narrow phase.
     */
    void addCollisionModel (core::CollisionModel *cm) override;

    static bool keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2);

    /// Bounding tree is not required by this detection algorithm
    bool needsDeepBoundingTree() const override { return false; }

protected:

    /// Return true if the provided CollisionModel can collide with itself
    bool doesSelfCollide(core::CollisionModel *cm) const;

    /// Return true if the provided CollisionModel intersect boxModel, false otherwise
    bool intersectWithBoxModel(core::CollisionModel *cm) const;

    collision::geometry::CubeCollisionModel::SPtr boxModel;

    /// A data structure to store a pair of collision models
    /// They both describe the same object
    struct FirstLastCollisionModel
    {
        /// First collision model in the hierarchy of collision models of an object. Usually a bounding box
        core::CollisionModel* firstCollisionModel { nullptr };

        // Last collision model in the hierarchy of collision models of an object. Holding more details than a bounding box
        core::CollisionModel* lastCollisionModel { nullptr };

        FirstLastCollisionModel(core::CollisionModel* a, core::CollisionModel* b) : firstCollisionModel(a), lastCollisionModel(b) {}
    };

    /// vector of accumulated CollisionModel's when the collision pipeline asks
    /// to add a CollisionModel in BruteForceBroadPhase::addCollisionModel
    /// This vector is emptied at each time step in BruteForceBroadPhase::beginBroadPhase
    sofa::type::vector<FirstLastCollisionModel> m_collisionModels;
};

}
