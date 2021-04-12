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
#include <SofaBaseCollision/config.h>

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/helper/vector.h>

#include <queue>
#include <stack>

namespace sofa::core::collision
{
    class ElementIntersector;
}

namespace sofa::component::collision
{

class MirrorIntersector;

class SOFA_SOFABASECOLLISION_API BruteForceDetection :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
    /// Range defined by two iterators in a container of CollisionElement
    using CollisionIteratorRange = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>;

    /// A pair of CollisionIteratorRange where the first range contains elements from a first collision model
    /// and the second range contains collision elements from a second collision model
    /// This type is used when testing collision between two collision models
    /// Note that the second collision model can be the same than the first in case of self collision
    using TestPair = std::pair< CollisionIteratorRange, CollisionIteratorRange >;

public:
    SOFA_CLASS2(BruteForceDetection, core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

private:

    /// vector of accumulated CollisionModel's when the collision pipeline asks
    /// to add a CollisionModel in BruteForceDetection::addCollisionModel
    /// This vector is emptied at each time step in BruteForceDetection::beginBroadPhase
    sofa::helper::vector<core::CollisionModel*> collisionModels;

    Data< helper::fixed_array<sofa::defaulttype::Vector3,2> > box; ///< if not empty, objects that do not intersect this bounding-box will be ignored

    CubeCollisionModel::SPtr boxModel;


protected:
    BruteForceDetection();

    ~BruteForceDetection() override = default;

    virtual bool keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2);

public:

    void init() override;
    void reinit() override;

    //////////////////////////////
    /// BROAD PHASE interface ///
    //////////////////////////////

    /** \brief In the broad phase, ignores collision with the provided collision model if possible and add pairs of
     * collision models if in intersection.
     *
     * Ignore the collision with the provided collision model if it does not intersect with the box defined in
     * the Data box when it is defined.
     * Add the provided collision model to be investigated in the narrow phase in case of self collision.
     * Check intersection with already added collision models. If it can intersect another collision model, the pair
     * is added to be further investigated in the narrow phase.
     */
    void addCollisionModel(core::CollisionModel *cm) override;

    void beginBroadPhase() override;
    void endBroadPhase() override;

    //////////////////////////////
    /// NARROW PHASE interface ///
    //////////////////////////////

    /** \brief In the narrow phase, examine a potential collision between a pair of collision models, which has
     * been detected in the broad phase.
     *
     * The function traverses the hierarchy of CollisionElement's contained in each CollisionModel to avoid
     * unnecessary pair intersections.
     * An iterative form of the hierarchy traversal is adopted instead of a recursive form.
     */
    void addCollisionPair(const std::pair<core::CollisionModel*, core::CollisionModel*>& cmPair) override;


    void draw(const core::visual::VisualParams* /* vparams */) override { }

    inline bool needsDeepBoundingTree() const override { return true; }

protected:

    /// Return true if the provided CollisionModel intersect boxModel, false otherwise
    bool intersectWithBoxModel(core::CollisionModel *cm) const;

    /// Return true if the provided CollisionModel can collide with itself
    bool doesSelfCollide(core::CollisionModel *cm) const;

    /// Return true if both collision models belong to the same object, false otherwise
    static bool isSelfCollision(core::CollisionModel* cm1, core::CollisionModel* cm2);

    /// Build a list of TestPair's from internal and external children of two CollisionModel's
    static void initializeExternalCells(
            core::CollisionModel *cm1,
            core::CollisionModel *cm2,
            std::queue<TestPair>& externalCells);

    /// Store data related to two finnest CollisionModel's
    struct FinnestCollision
    {
        core::CollisionModel* cm1 { nullptr };
        core::CollisionModel* cm2 { nullptr };

        /// ElementIntersector corresponding to cm1 and cm2
        core::collision::ElementIntersector* intersector { nullptr };

        // True in case cm1 and cm2 belong to the same object, false otherwise
        bool selfCollision { false };
    };

    void processExternalCell(const TestPair &externalCell,
                             core::CollisionModel *&cm1,
                             core::CollisionModel *&cm2,
                             core::collision::ElementIntersector *coarseIntersector,
                             const FinnestCollision &finnest,
                             MirrorIntersector *mirror,
                             std::queue<TestPair> &externalCells,
                             sofa::core::collision::DetectionOutputVector *&outputs) const;

    static void
    processInternalCell(const TestPair &internalCell,
                        core::collision::ElementIntersector *coarseIntersector,
                        const FinnestCollision &finnest,
                        std::queue<TestPair> &externalCells,
                        std::stack<TestPair> &internalCells,
                        sofa::core::collision::DetectionOutputVector *&outputs);

    static void visitCollisionElements(const TestPair &root,
                                       core::collision::ElementIntersector *coarseIntersector,
                                       const FinnestCollision &finnest,
                                       std::queue<TestPair> &externalCells,
                                       std::stack<TestPair> &internalCells,
                                       sofa::core::collision::DetectionOutputVector *&outputs);

    static void
    visitExternalChildren(const core::CollisionElementIterator &it1, const core::CollisionElementIterator &it2,
                          core::collision::ElementIntersector *coarseIntersector,
                          const FinnestCollision &finnest,
                          std::queue<TestPair> &externalCells,
                          sofa::core::collision::DetectionOutputVector *&outputs);

    /// Test intersection between two ranges of CollisionElement's
    /// The provided TestPair contains ranges of external CollisionElement's, which means that
    /// they can be tested against each other for intersection
    static void finalCollisionPairs(const TestPair& pair,
                             bool selfCollision,
                             core::collision::ElementIntersector* intersector,
                             sofa::core::collision::DetectionOutputVector*& outputs);

private:

    /// Get both collision models corresponding to the provided TestPair
    static std::pair<core::CollisionModel*, core::CollisionModel*> getCollisionModelsFromTestPair(const TestPair& pair);

    static bool isRangeEmpty(const CollisionIteratorRange& range);
};

} // namespace sofa::component::collision
