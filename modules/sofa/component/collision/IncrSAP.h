/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INCRSAP_H
#define INCRSAP_H

#include <sofa/core/collision/BroadPhaseDetection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/core/CollisionElement.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/component/component.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/component/collision/EndPoint.h>
#include <set>
#include <map>
#include <deque>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/CollisionPM.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa
{

namespace component
{

namespace collision
{

class EndPointID;

class ISAPBox{
public:
    ISAPBox(){}

    ISAPBox(Cube c) : cube(c){}

    bool overlaps(const ISAPBox & other,int axis)const;//we use here end points
    bool overlaps(const ISAPBox & other)const;//we use min and max vect of the field cube

    inline void show()const{
        std::cout<<"MIN "<<cube.minVect()<<std::endl;
        std::cout<<"MAX "<<cube.maxVect()<<std::endl;
    }

    bool moving(int axis)const;

    bool moving()const;

    void init(int boxID,EndPointID ** endPts);

    void update();

    const core::CollisionElementIterator finalElement()const;

    EndPointID & min(int dim);
    const EndPointID & min(int dim)const;
    EndPointID & max(int dim);
    const EndPointID & max(int dim)const;

    Cube cube;
    EndPointID * _min[3];
    EndPointID * _max[3];

    static double tolerance;
};


using namespace sofa::defaulttype;

template <template<class T,class Allocator> class List,template <class T> class Allocator = std::allocator>
class TIncrSAP :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(TIncrSAP,List,Allocator), core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

    typedef ISAPBox SAPBox;
    typedef List<EndPointID*,Allocator<EndPointID*> > EndPointList;

private:
    //void
    int greatestVarianceAxis()const;

    bool added(core::CollisionModel * cm)const;

    bool add(core::CollisionModel * cm);

    /**
      *Updates values of end points. These values are coordinates of AABB on axis that maximazes the variance for the AABBs.
      */
    void updateEndPoints();
    void setEndPointsID();


    void boxPrune();
    void updateMovingBoxes();
    void addIfCollide(int boxID1,int boxID2,int axis1,int axis2);
    void removeCollision(int a,int b);
    void reinitDetection();
    void purge();


    Data<bool> bDraw;

    Data< helper::fixed_array<Vector3,2> > box;

    CubeModel::SPtr boxModel;

    std::vector<ISAPBox> _boxes;
    EndPointList _end_points[3];
    CollidingPM _colliding_elems;

    int _cur_axis;
    bool _nothing_added;

    std::set<core::CollisionModel*> collisionModels;
protected:
    TIncrSAP();

    virtual ~TIncrSAP();

public:
    void setDraw(bool val) { bDraw.setValue(val); }

    void init();
    void reinit();

    void addCollisionModel (core::CollisionModel *cm);

    /**
      *Unuseful methods because all is done in addCollisionModel
      */
    void addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& ){}
    void addCollisionPairs (std::vector<std::pair<core::CollisionModel*, core::CollisionModel*> >&){}

    virtual void beginNarrowPhase();


    /* for debugging */
    inline void draw(const core::visual::VisualParams*){}

    inline virtual bool needsDeepBoundingTree()const{return false;}
};

typedef TIncrSAP<std::vector,std::allocator> IncrSAP;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_MESH_COLLISION)
extern template class SOFA_MESH_COLLISION_API TIncrSAP<helper::vector,helper::CPUMemoryManager>;
extern template class SOFA_MESH_COLLISION_API TIncrSAP<std::vector,std::allocator>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif // INCRSAP_H
