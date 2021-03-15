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

#include <SofaGeneralMeshCollision/config.h>

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaMeshCollision/EndPoint.h>
#include <sofa/defaulttype/Vec.h>
#include <set>

#include "sofa/helper/ScopedAdvancedTimer.h"

namespace sofa::core::objectmodel
{
    class BaseContext;
}

namespace sofa::core::collision
{
    class ElementIntersector;
}

namespace sofa::component::collision
{

class EndPoint;

/**
  *SAPBox is a simple bounding box. It contains a Cube which contains only one final
  *CollisionElement and pointers to min and max EndPoints. min and max end points
  *are respectively min and max coordinates of the cube on a coordinate axis.
  *min and max are updated with the method update(int i), so min and max have
  *min/max values on the i-th axis after the method update(int i).
  */
class SOFA_SOFAGENERALMESHCOLLISION_API DSAPBox{
public:
    explicit DSAPBox(const Cube& c,EndPoint * mi = nullptr,EndPoint * ma = nullptr) : cube(c),min(mi),max(ma){}

    void update(int axis,double alarmDist);

    [[nodiscard]]
    double squaredDistance(const DSAPBox & other) const;

    /// Compute the squared distance from this to other on a specific axis
    [[nodiscard]]
    double squaredDistance(const DSAPBox & other, int axis)const;

    void show() const;

    Cube cube;
    EndPoint * min { nullptr };
    EndPoint * max { nullptr };
};

/**
  *This class is an implementation of sweep and prune in its "direct" version, i.e. at each step
  *it sorts all the primitives along an axis (not checking the moving ones) and computes overlaping pairs without
  *saving it. But the memory used to save these primitives is created just once, the first time we add CollisionModels.
  */
class SOFA_SOFAGENERALMESHCOLLISION_API DirectSAP :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(DirectSAP, core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

    typedef sofa::helper::vector<EndPoint*> EndPointList;
    typedef DSAPBox SAPBox;

private:

    /** \brief Returns the axis number which have the greatest variance for the primitive end points.
     *
     * This axis is used when updating and sorting end points. The greatest variance means
     * that this axis have the most chance to eliminate a maximum of not overlaping SAPBox pairs
     * because along this axis, SAPBoxes are the sparsest.
     */
    int greatestVarianceAxis()const;

    /// Return true if the collision model has already been added to the list of managed models
    bool added(core::CollisionModel * cm) const;

    /// Add a collision model to the list of managed models
    void add(core::CollisionModel * cm);

    /**
      * Updates values of end points. These values are coordinates of AABB on axis that maximize the variance for the AABBs.
      */
    void update();

    Data<bool> d_draw; ///< enable/disable display of results
    Data<bool> d_showOnlyInvestigatedBoxes;
    Data<int> d_nbPairs; ///< number of pairs of elements sent to narrow phase
    Data< helper::fixed_array<defaulttype::Vector3,2> > d_box; ///< if not empty, objects that do not intersect this bounding-box will be ignored

    CubeCollisionModel::SPtr m_boxModel;

    /// Store a permanent list of end points
    /// The container is a std::list to avoid invalidation of pointers after an insertion
    std::list<EndPoint> m_endPointContainer;

    sofa::helper::vector<DSAPBox> m_boxes;//boxes
    sofa::helper::vector<bool> m_isBoxInvestigated;
    EndPointList m_sortedEndPoints; ///< list of EndPoints dedicated to be sorted. Owner of pointers is m_endPointContainer
    int m_currentAxis;//the current greatest variance axis

    std::set<core::CollisionModel*> m_collisionModels;//used to check if a collision model is added
    sofa::helper::vector<core::CollisionModel*> m_newCollisionModels;//eventual new collision models to add at a step

    double m_alarmDist;
    double m_alarmDist_d2;
    double m_sq_alarmDist;

    static bool isSquaredDistanceLessThan(const DSAPBox& a, const DSAPBox& b, double threshold);

protected:
    DirectSAP();

    ~DirectSAP() override = default;

    struct BoxData
    {
        core::CollisionModel* collisionModel { nullptr };
        sofa::core::objectmodel::BaseContext* context { nullptr };
        bool isBoxSimulated { false };
        bool doesBoxSelfCollide { false };
        sofa::core::CollisionElementIterator collisionElementIterator;
    };
    std::vector<BoxData> m_boxData;

    bool isPairFiltered(const BoxData& data0, const BoxData& data1,
                        const DSAPBox& box0, int boxId1
                ) const;

    void narrowCollisionDetectionForPair(
            core::collision::ElementIntersector* intersector,
            core::CollisionModel *collisionModel0,
            core::CollisionModel *collisionModel1,
            core::CollisionElementIterator collisionModelIterator0,
            core::CollisionElementIterator collisionModelIterator1);

    void createBoxesFromCollisionModels();
    void cacheData(); /// Cache data into vector to avoid overhead during access
    void sortEndPoints();
    void narrowCollisionDetectionFromSortedEndPoints();

public:
    void setDraw(bool val) { d_draw.setValue(val); }

    void init() override;
    void reinit() override;
    void reset() override;

    void addCollisionModel (core::CollisionModel *cm) override;

    /**
      *Unuseful methods because all is done in addCollisionModel
      */
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& ) override {}
    void addCollisionPairs (const helper::vector<std::pair<core::CollisionModel*, core::CollisionModel*> >&) override {}

    void endBroadPhase() override;
    void beginNarrowPhase() override;

    /* for debugging */
    void draw(const core::visual::VisualParams*) override;

    inline bool needsDeepBoundingTree()const override {return false;}

};

} // namespace sofa::component::collision
