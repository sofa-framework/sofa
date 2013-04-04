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

namespace sofa
{

namespace component
{

namespace collision
{

class EndPointID;

class SAPBox{
public:
    SAPBox(Cube c) : cube(c){}

    void update(int axis);

    bool overlaps(const SAPBox & other,int axis)const;

    bool overlaps(const SAPBox &other)const;

    inline void show()const{
        std::cout<<"MIN "<<cube.minVect()<<std::endl;
        std::cout<<"MAX "<<cube.maxVect()<<std::endl;
    }

    bool moving(int axis)const;

    void init(int boxID = -1);

    Cube cube;
    EndPointID * min[3];
    EndPointID * max[3];
};


using namespace sofa::defaulttype;

template <template<class T,class Allocator> class List,template <class T> class Allocator = std::allocator>
class SOFA_BASE_COLLISION_API TIncrSAP :
    public core::collision::BroadPhaseDetection,
    public core::collision::NarrowPhaseDetection
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE2(TIncrSAP,List,Allocator), core::collision::BroadPhaseDetection, core::collision::NarrowPhaseDetection);

    typedef List<EndPoint*,Allocator<EndPoint*> > EndPointList;

private:
    //void
    int greatestVarianceAxis()const;

    bool added(core::CollisionModel * cm)const;

    void add(core::CollisionModel * cm);

    /**
      *Updates values of end points. These values are coordinates of AABB on axis that maximazes the variance for the AABBs.
      */
    void update();

    Data<bool> bDraw;

    Data< helper::fixed_array<Vector3,2> > box;

    CubeModel::SPtr boxModel;

    std::vector<SAPBox> _boxes;
    EndPointList _end_points[3];
    int _cur_axis;

    std::set<core::CollisionModel*> collisionModels;
protected:
    TIncrSAP();

    helper::vector<Cube> & cubes(const CubeModel* cm);

    ~TIncrSAP();
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

typedef TIncrSAP<std::vector,std::allocator> DirectSAP;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_COLLISION)
extern template class SOFA_BASE_COLLISION_API TIncrSAP<helper::vector,helper::CPUMemoryManager>;
extern template class SOFA_BASE_COLLISION_API TIncrSAP<std::vector,std::allocator>;
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif // INCRSAP_H
