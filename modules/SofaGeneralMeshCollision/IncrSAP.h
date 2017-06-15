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
#ifndef SOFA_COMPONENT_COLLISION_INCRSAP_H
#define SOFA_COMPONENT_COLLISION_INCRSAP_H
#include "config.h"

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/CollisionModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/defaulttype/Vec.h>
#include <SofaMeshCollision/EndPoint.h>
#include <set>
#include <map>
#include <deque>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/CollisionPM.h>
#include <sofa/helper/AdvancedTimer.h>
#include <vector>

namespace sofa
{

namespace component
{

namespace collision
{

class EndPointID;

/**
  *ISAPBox is a simple bounding box. It contains a Cube which contains only one final
  *CollisionElement and pointers to min and max EndPoints along the three dimensions. min and max end points
  *are respectively min and max coordinates of the cube on a coordinate axis.
  *The between end poinsts (_min, _max) and the field cube is that cube is always updated whereas
  *_min and _max are stored values of the cube end points at previous time step.
  */
class SOFA_GENERAL_MESH_COLLISION_API ISAPBox{
public:
    ISAPBox(){}

    ISAPBox(Cube c) : cube(c){}

    /**
      *Returns true if this overlaps other along the dimension axis.
      *For the two following methods, end points are not used but real positions
      *of end points of the field cube.
      */
    bool endPointsOverlap(const ISAPBox & other,int axis)const;

    /**
      *Returns true if this overlaps other along the three dimensions.
      */
    bool overlaps(const ISAPBox & other,double alarmDist)const;

    double squaredDistance(const ISAPBox & other)const;

    inline void show()const{
        msg_info("IncrSAP") <<"MIN "<<cube.minVect() ;
        msg_info("IncrSAP") <<"MAX "<<cube.maxVect() ;
    }

    inline void showEndPoints()const{
        std::stringstream tmp;
        tmp<<"MIN ";
        for(int i = 0 ; i < 3 ; ++i)
            tmp<<min(i).value<<" ";
        tmp<<msgendl;

        tmp<<"MAX ";
        for(int i = 0 ; i < 3 ; ++i)
            tmp<<max(i).value<<" ";
        msg_info("IncrSAP") << tmp.str() ;
    }

    /**
      *Returns true if the ISAPBox is moving along the dimension axis. i.e., returns true if the value of the end point of dimension axis is different
      *from the end point of the field cube (which is the real position of the ISAPBox).
      */
    bool moving(int axis,double alarmDist)const;

    /**
      *The same than the previous one except that this one checks the three dimensions, i.e. it returns true if
      *the ISAPBox is moving at least along one dimension.
      */
    bool moving(double alarmDist)const;

    /**
      *Inits _min and _max fiels with endPts. endPts is an one dimension array of EndPointID pointers.
      *After this method, the first three end points are the mins in the dimension 0, 1, 2.
      *The last three end points are the maxs in the dimension 0, 1, 2.
      *Values and IDs of endPts are updated after this method.
      */
    void init(int boxID,EndPointID ** endPts);

    void update(double alarmDist);

    void updatedMin(int dim,EndPointID &end_point,double alarmDist)const;
    void updatedMax(int dim,EndPointID &end_point,double alarmDist)const;


    void updateMin(int dim,double alarmDist);
    void updateMax(int dim,double alarmDist);

    bool minMoving(int axis,double alarmDist) const;
    bool maxMoving(int axis,double alarmDist) const;

    const core::CollisionElementIterator finalElement()const;

    EndPointID & min(int dim);
    const EndPointID & min(int dim)const;
    EndPointID & max(int dim);
    const EndPointID & max(int dim)const;

    double curMin(int dim) const;
    double curMax(int dim)const;

    // Returns true if the endpoints have id ID and if min end points are min and max are max.
    // It checks only the field data.
    bool endPointsAreAlright(int ID);

    Cube cube;
    EndPointID * _min[3];
    EndPointID * _max[3];

    static double tolerance;
};

/**
  *Implementation of incremental sweep and prune. i.e. collision are stored and updated which should speed up
  *the collision detection compared to the DirectSAP.
  */
class SOFA_GENERAL_MESH_COLLISION_API IncrSAP :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(IncrSAP, core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

    typedef ISAPBox SAPBox;
    typedef std::vector<EndPointID*> EndPointList;

private:
    /**
      *Returns the dimension number for which one have the greatest variance of end points position.
      */
    int greatestVarianceAxis()const;

    bool added(core::CollisionModel * cm)const;

    bool add(core::CollisionModel * cm);

    /**
      *Updates values of end points. These values are coordinates of AABB on axis that maximazes the variance for the AABBs.
      */
    void updateEndPoints();

    /**
      *Sets the end points ID, i.e. each end point in the list after this mehod have its position (ID) updated.
      */
    void setEndPointsID();


    /**
      *A counterpart of DirectSAP which is used when a new collision model is added to the IncrSAP. It is more efficient than
      *updating every box added to the IncrSAP.
      */
    void boxPrune();

    /**
      *When there is no added collision model, one update only the moving boxes and in the same time, the collisions.
      */
    void updateMovingBoxes();

    /**
      *Checks that boxes whose IDs are boxID1 and boxID2 are in collision, and add it to the list of collisions.
      */
    void addIfCollide(int boxID1,int boxID2);

    /**
      *Checks that boxes whose IDs are boxID1 and boxID2 are in collision along axes axis1 and axis2, and add it to the list of collisions.
      */
    void addIfCollide(int boxID1,int boxID2,int axis1,int axis2);
    void removeCollision(int a,int b);
    void reinitDetection();

    /**
      *Used in initialisatio of IncrSAP. It clears all the IncrSAP fields.
      */
    void purge();


    Data<bool> bDraw;

    Data< helper::fixed_array<defaulttype::Vector3,2> > box;

    CubeModel::SPtr boxModel;

    std::vector<ISAPBox> _boxes;
    EndPointList _end_points[3];
    CollidingPM _colliding_elems;


    //The following methods are used when updating end points in the end point lists, it updates in the same time the collisions.
    void moveMinForward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & next_it);
    void moveMaxForward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & next_it);
    void moveMinBackward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & prev_it);
    void moveMaxBackward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & prev_it);

    static bool assertion_order(EndPointList::iterator it,EndPointList::iterator begin,EndPointList::iterator end);
    static bool assertion_list_order(EndPointList::iterator begin_it,const EndPointList::iterator & end_it);
    static bool assertion_superior(EndPointList::iterator begin_it,const EndPointList::iterator & end_it,EndPoint* point);
    static bool assertion_inferior(EndPointList::iterator begin_it,const EndPointList::iterator & end_it,EndPoint* point);
    bool assertion_end_points_sorted()const;
    //EndPointID & findEndPoint(int dim,int data);



    int _cur_axis;
    bool _nothing_added;
    double _alarmDist;
    double _alarmDist_d2;


    std::set<core::CollisionModel*> collisionModels;
protected:
    IncrSAP();

    virtual ~IncrSAP();

public:
    void setDraw(bool val) { bDraw.setValue(val); }

    void init();
    void reinit();

    void addCollisionModel (core::CollisionModel *cm);

    /**
      *Unuseful methods because all is done in addCollisionModel
      */
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& ){}
    void addCollisionPairs (const helper::vector<std::pair<core::CollisionModel*, core::CollisionModel*> >&){}

    virtual void beginNarrowPhase();


    /* for debugging */
    inline void draw(const core::visual::VisualParams*){}

    inline virtual bool needsDeepBoundingTree()const{return false;}

    void showEndPoints()const;

    void showBoxes()const;
};


} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_INCRSAP_H
