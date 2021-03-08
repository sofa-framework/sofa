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

    double squaredDistance(const DSAPBox & other) const;

    /// Compute the squared distance from this to other on a specific axis
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

    typedef std::vector<EndPoint*> EndPointList;

    typedef DSAPBox SAPBox;

    //void collidingCubes(std::vector<std::pair<Cube,Cube> > & col_cubes)const;
private:
    /**
      *Returns the axis number which have the greatest variance for the primitive end points.
      *This axis is used when updating and sorting end points. The greatest variance means
      *that this axis have the most chance to eliminate a maximum of not overlaping SAPBox pairs
      *because along this axis, SAPBoxes are the sparsest.
      */
    int greatestVarianceAxis()const;

    bool added(core::CollisionModel * cm)const;

    void add(core::CollisionModel * cm);

    /**
      *Updates values of end points. These values are coordinates of AABB on axis that maximazes the variance for the AABBs.
      */
    void update();

    Data<bool> bDraw; ///< enable/disable display of results

    Data< helper::fixed_array<defaulttype::Vector3,2> > box; ///< if not empty, objects that do not intersect this bounding-box will be ignored

    CubeCollisionModel::SPtr boxModel;

    std::vector<DSAPBox> _boxes;//boxes
    EndPointList _end_points;//end points of _boxes
    int _cur_axis;//the current greatest variance axis

    std::set<core::CollisionModel*> collisionModels;//used to check if a collision model is added
    std::vector<core::CollisionModel*> _new_cm;//eventual new collision models to  add at a step

    double _alarmDist;
    double _alarmDist_d2;
    double _sq_alarmDist;
protected:
    DirectSAP();

    ~DirectSAP() override;

    std::vector<EndPoint*> _to_del;//EndPoint arrays to delete when deleting DirectSAP
public:
    void setDraw(bool val) { bDraw.setValue(val); }

    void init() override;
    void reinit() override;

    void addCollisionModel (core::CollisionModel *cm) override;

    /**
      *Unuseful methods because all is done in addCollisionModel
      */
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& ) override {}
    void addCollisionPairs (const helper::vector<std::pair<core::CollisionModel*, core::CollisionModel*> >&) override {}

    void endBroadPhase() override;
    void beginNarrowPhase() override;


    /* for debugging */
    void draw(const core::visual::VisualParams*) override {}

    inline bool needsDeepBoundingTree()const override {return false;}
};

} // namespace sofa::component::collision
