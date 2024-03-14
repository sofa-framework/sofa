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

#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/component/collision/detection/algorithm/EndPoint.h>
#include <sofa/component/collision/detection/algorithm/DSAPBox.h>
#include <unordered_set>

namespace sofa::core::collision
{
class ElementIntersector;
}

namespace sofa::component::collision::detection::algorithm
{

/**
 * This class is an implementation of sweep and prune in its "direct" version, i.e. at each step
 * it sorts all the primitives along an axis (not checking the moving ones) and computes overlaping pairs without
 * saving it. But the memory used to save these primitives is created just once, the first time we add CollisionModels.
 */
class SOFA_COMPONENT_COLLISION_DETECTION_ALGORITHM_API DirectSAPNarrowPhase : public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS(DirectSAPNarrowPhase, core::collision::NarrowPhaseDetection);

    typedef sofa::type::vector<EndPoint *> EndPointList;

private:

    /** \brief Returns the axis number which have the greatest variance for the primitive end points.
     *
     * This axis is used when updating and sorting end points. The greatest variance means
     * that this axis have the most chance to eliminate a maximum of not overlaping SAPBox pairs
     * because along this axis, SAPBoxes are the sparsest.
     */
    int greatestVarianceAxis() const;

    /**
      * Updates values of end points. These values are coordinates of AABB on axis that maximize the variance for the AABBs.
      */
    void updateBoxes();

    SOFA_ATTRIBUTE_DEPRECATED__DRAWNARROWPHASE()
    sofa::core::objectmodel::lifecycle::DeprecatedData d_draw{this, "v23.12", "v24.06", "draw", "Use display flag 'showDetectionOutputs' instead"}; ///< enable/disable display of results
    Data<bool> d_showOnlyInvestigatedBoxes; ///< Show only boxes which will be sent to narrow phase
    Data<int> d_nbPairs; ///< number of pairs of elements sent to narrow phase

    /// Store a permanent list of end points
    /// The container is a std::list to avoid invalidation of pointers after an insertion
    std::list<EndPoint> m_endPointContainer;

    sofa::type::vector<DSAPBox> m_boxes;//boxes
    sofa::type::vector<bool> m_isBoxInvestigated;
    EndPointList m_sortedEndPoints; ///< list of EndPoints dedicated to be sorted. Owner of pointers is m_endPointContainer
    int m_currentAxis;//the current greatest variance axis

    std::unordered_set<core::CollisionModel *> m_addedCollisionModels;//used to check if a collision model is added
    sofa::type::vector<core::CollisionModel *> m_newCollisionModels;//eventual new collision models to add at a step

    double m_alarmDist;
    double m_alarmDist_d2;
    double m_sq_alarmDist;

    static bool isSquaredDistanceLessThan(const DSAPBox &a, const DSAPBox &b, double threshold);

protected:
    DirectSAPNarrowPhase();

    ~DirectSAPNarrowPhase() override = default;

    std::unordered_set<core::CollisionModel *> m_broadPhaseCollisionModels;

    struct BoxData
    {
        core::CollisionModel *lastCollisionModel{nullptr};
        sofa::core::objectmodel::BaseContext *context{nullptr};
        bool isBoxSimulated{false};
        bool doesBoxSelfCollide{false};
        sofa::core::CollisionElementIterator collisionElementIterator;
        bool isInBroadPhase{false};
    };
    std::vector<BoxData> m_boxData;

    bool isPairFiltered(const BoxData &data0, const BoxData &data1, const DSAPBox &box0, int boxId1) const;

    void narrowCollisionDetectionForPair(
            core::collision::ElementIntersector *intersector,
            core::CollisionModel *collisionModel0,
            core::CollisionModel *collisionModel1,
            core::CollisionElementIterator collisionModelIterator0,
            core::CollisionElementIterator collisionModelIterator1);

    void createBoxesFromCollisionModels();

    void cacheData(); /// Cache data into vector to avoid overhead during access
    void sortEndPoints();

    void narrowCollisionDetectionFromSortedEndPoints();

public:

    void reset() override;

    void beginNarrowPhase() override;

    void addCollisionPair(const std::pair<core::CollisionModel *, core::CollisionModel *> &cmPair) override;

    void endNarrowPhase() override;

    /* for debugging */
    void draw(const core::visual::VisualParams *) override;

    /// Get the result of the broad phase and check if there are some new collision models that was not yet processed
    void checkNewCollisionModels();

    /// Bounding tree is not required by this detection algorithm
    bool needsDeepBoundingTree() const override { return false; }
};

}