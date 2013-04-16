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
#ifndef INCRSAP_INL
#define INCRSAP_INL
#include <sofa/component/collision/IncrSAP.h>

namespace sofa
{
namespace component
{
namespace collision
{

inline EndPointID & ISAPBox::min(int dim){return *_min[dim];}
inline const EndPointID & ISAPBox::min(int dim)const{return *_min[dim];}

inline EndPointID & ISAPBox::max(int dim){return *_max[dim];}
inline const EndPointID & ISAPBox::max(int dim)const{return *_max[dim];}

inline void ISAPBox::update(){
    for(int i = 0 ; i < 3 ; ++i){
        _min[i]->value = cube.minVect()[i];
        _max[i]->value = cube.maxVect()[i];
    }
}

inline void ISAPBox::init(int boxID,EndPointID ** endPts){
    for(int i = 0 ; i < 3 ; ++i){
        _min[i] = endPts[i];
        _max[i] = endPts[3 + i];
    }

    for(int i = 0 ; i < 3 ; ++i){
        _min[i]->setBoxID(boxID);
        _max[i]->setBoxID(boxID);
        _min[i]->setMin();
        _max[i]->setMax();
    }    

    //update();
}

inline bool ISAPBox::overlaps(const ISAPBox & other, int axis) const{
    assert(axis >= 0);
    assert(axis < 3);
    if(((this->_min[axis])->value >= other._max[axis]->value) || (other._min[axis]->value >= this->_max[axis]->value))
        return false;

    return true;
}

inline bool ISAPBox::moving(int axis) const{
    const core::CollisionElementIterator & finE = finalElement();

    const core::CollisionModel * cm = finE.getCollisionModel();

    switch (cm->getEnumType()){
        case core::CollisionModel::OBB_TYPE:
            return fabs(((static_cast<const OBBModel*>(cm))->lvelocity(finE.getIndex()))[axis]) > tolerance;
            break;
        case core::CollisionModel::CAPSULE_TYPE:
            return fabs(((static_cast<const CapsuleModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
            break;
        case core::CollisionModel::SPHERE_TYPE:
            return fabs(((static_cast<const SphereModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
            break;
        case core::CollisionModel::TRIANGLE_TYPE:
            return fabs(((static_cast<const TriangleModel*>(cm))-> velocity(finE.getIndex()))[axis]) > tolerance;
            break;
        case core::CollisionModel::LINE_TYPE:
            return fabs(((static_cast<const LineModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
            break;
        case core::CollisionModel::POINT_TYPE:
            return fabs(((static_cast<const PointModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
            break;
        default:
            std::cerr<<"CollisionModel type not found within SAPBox::moving"<<std::endl;
            return true;
    }
}

inline bool ISAPBox::moving() const{
    const core::CollisionElementIterator & finE = finalElement();

    const core::CollisionModel * cm = finE.getCollisionModel();

    switch (cm->getEnumType()){
        case core::CollisionModel::OBB_TYPE:
            return ((static_cast<const OBBModel*>(cm))->lvelocity(finE.getIndex())).norm2() > tolerance*tolerance;
            break;
        case core::CollisionModel::CAPSULE_TYPE:
            return ((static_cast<const CapsuleModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
            break;
        case core::CollisionModel::SPHERE_TYPE:
            return ((static_cast<const SphereModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
            break;
        case core::CollisionModel::TRIANGLE_TYPE:
            return ((static_cast<const TriangleModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
            break;
        case core::CollisionModel::LINE_TYPE:
            return ((static_cast<const LineModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
            break;
        case core::CollisionModel::POINT_TYPE:
            return ((static_cast<const PointModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
            break;
        default:
            std::cerr<<"CollisionModel type not found within SAPBox::moving"<<std::endl;
            return true;
    }
}

inline const core::CollisionElementIterator ISAPBox::finalElement()const{
    return cube.getExternalChildren().first;
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
TIncrSAP<List,Allocator>::TIncrSAP()
    : bDraw(initData(&bDraw, false, "draw", "enable/disable display of results"))
    , box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored")),
      _nothing_added(true)
{
    //_end_points = new EndPointList[3];
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
TIncrSAP<List,Allocator>::~TIncrSAP(){
    for(int i = 0 ; i < 3 ; ++i)
        for(typename EndPointList::iterator it = _end_points[i].begin() ; it != _end_points[i].end() ; ++it)
            delete (*it);


    //delete[] _end_points;
}


template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::purge(){
    for(int i = 0 ; i < 3 ; ++i){
        for(typename EndPointList::iterator it = _end_points[i].begin() ; it != _end_points[i].end() ; ++it)
            delete (*it);

        _end_points[i].clear();
    }

    _boxes.clear();
    _colliding_elems.clear();
    collisionModels.clear();
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::init()
{
    reinit();
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::reinit()
{
    if (box.getValue()[0][0] >= box.getValue()[1][0])
    {
        boxModel.reset();
    }
    else
    {
        if (!boxModel) boxModel = sofa::core::objectmodel::New<CubeModel>();
        boxModel->resize(1);
        boxModel->setParentOf(0, box.getValue()[0], box.getValue()[1]);
    }

    purge();
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
inline bool TIncrSAP<List,Allocator>::added(core::CollisionModel *cm) const
{
    return collisionModels.count(cm->getLast()) >= 1;
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
inline bool TIncrSAP<List,Allocator>::add(core::CollisionModel *cm)
{
    return (collisionModels.insert(cm->getLast())).second;
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
inline void TIncrSAP<List,Allocator>::addCollisionModel(core::CollisionModel *cm)
{
    if(add(cm)){
        _nothing_added = false;

        CubeModel * cube_model = dynamic_cast<CubeModel *>(cm->getLast()->getPrevious());

        int old_size = _boxes.size();
        int cube_model_size = cube_model->getSize();
        _boxes.resize(cube_model_size + old_size);

        EndPointID * endPts[6];
        for(int i = 0 ; i < cube_model->getSize() ; ++i){
            for(int j = 0 ; j < 6 ; ++j)
                endPts[j] = new EndPointID;

            ISAPBox & new_box = _boxes[old_size + i];
            new_box.cube = Cube(cube_model,i);
            new_box.init(i + old_size,endPts);

            for(int j = 0 ; j < 3 ; ++j){
                _end_points[j].push_back(&(new_box.min(j)));
                _end_points[j].push_back(&(new_box.max(j)));
            }
        }
    }
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
int TIncrSAP<List,Allocator>::greatestVarianceAxis()const{
    double diff;
    double v[3];//variances for each axis
    double m[3];//means for each axis
    for(int i = 0 ; i < 3 ; ++i)
        v[i] = m[i] = 0;

    //computing the mean value of end points on each axis
    for(int j = 0 ; j < 3 ; ++j)
        for(typename EndPointList::const_iterator it = _end_points[j].begin() ; it != _end_points[j].end() ; ++it)
            m[j] += (**it).value;

    m[0] /= 2*_boxes.size();
    m[1] /= 2*_boxes.size();
    m[2] /= 2*_boxes.size();

    //computing the variance of end points on each axis
    for(int j = 0 ; j < 3 ; ++j){
        for(typename EndPointList::const_iterator it = _end_points[j].begin() ; it != _end_points[j].end() ; ++it){
            diff = (**it).value - m[j];
            v[j] += diff*diff;
        }
    }

    //std::cout<<"end greatestVarianceAxis"<<std::endl;

    if(v[0] >= v[1] && v[0] >= v[2])
        return 0;
    else if(v[1] >= v[2])
        return 1;
    else
        return 2;
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::updateEndPoints(){
    for(unsigned int i = 0 ; i < _boxes.size() ; ++i)
        _boxes[i].update();
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::setEndPointsID(){
    //std::cout<<"SETID"<<std::endl;
    for(int dim = 0 ; dim < 3 ; ++dim){
        int ID = 0;
        for(typename EndPointList::iterator it = _end_points[dim].begin() ; it != _end_points[dim].end() ; ++it){
            (**it).ID = ID++;
        }
    }
    //std::cout<<"SETID==========="<<std::endl;
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::reinitDetection(){
    //std::cout<<"REINIT"<<std::endl;
    _colliding_elems.clear();
    CompPEndPoint comp;
    for(int j = 0 ; j < 3 ; ++j){
        std::sort(_end_points[j].begin(),_end_points[j].end(),comp);
    }
    //std::cout<<"SORT SUCCESS"<<std::endl;
    setEndPointsID();
    //std::cout<<"REINIT============END"<<std::endl;
}


template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::addIfCollide(int boxID1,int boxID2,int axis1,int axis2){
    assert(boxID1 < (int)(_boxes.size()));
    assert(boxID2 < (int)(_boxes.size()));

    ISAPBox & box0 = _boxes[boxID1];
    ISAPBox & box1 = _boxes[boxID2];
    core::CollisionModel *finalcm1 = box0.cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
    core::CollisionModel *finalcm2 = box1.cube.getCollisionModel()->getLast();

//    if(!(finalcm1->isSimulated() || finalcm2->isSimulated()))
//        std::cout<<"not simulated"<<std::endl;
//    if(!((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2)))
//        std::cout<<"cannot collide"<<std::endl;
//    if(!box0.overlaps(box1,axis1))
//        std::cout<<"not overlapping by axis1 "<<axis1<<std::endl;
//    if(!box0.overlaps(box1,axis2))
//        std::cout<<"not overlapping by axis2 "<<axis2<<std::endl;

    if((finalcm1->isSimulated() || finalcm2->isSimulated()) &&
            (((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2)) && box0.overlaps(box1,axis1) && box0.overlaps(box1,axis2))){//intersection on all axes
        //sout << "Final phase "<<gettypename(typeid(*finalcm1))<<" - "<<gettypename(typeid(*finalcm2))<<sendl;
    //                    //std::cout<<"finalcm1 finalcm2 "<<finalcm1<<" "<<finalcm2<<std::endl;
    //                    //std::cout<<"intersectionMethod "<<intersectionMethod->getClass()->className<<std::endl;
    //                    //std::cout<<"Final phase "<<finalcm1->getClass()->className<<" - "<<finalcm2->getClass()->className<<std::endl;

        //std::cout<<"A TRUE COLLISION !!"<<std::endl;
        bool swapModels = false;
        core::collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(finalcm1, finalcm2, swapModels);//find the method for the finnest CollisionModels

        assert(box0.cube.getExternalChildren().first.getIndex() == box0.cube.getIndex());
        assert(box1.cube.getExternalChildren().first.getIndex() == box1.cube.getIndex());

        if((!swapModels) && finalcm1->getClass() == finalcm2->getClass() && finalcm1 > finalcm2)//we do that to have only pair (p1,p2) without having (p2,p1)
            swapModels = true;

        if(finalintersector != 0x0){
            if(swapModels){
                _colliding_elems.add(boxID1,boxID2,box1.finalElement(),box0.finalElement(),finalintersector);
            }
            else{
                _colliding_elems.add(boxID1,boxID2,box0.finalElement(),box1.finalElement(),finalintersector);
            }
        }
        else{
//                            std::cout<<"Final phase "<<finalcm1->getClass()->className<<" - "<<finalcm2->getClass()->className<<std::endl;
//                            std::cout<<"not found with intersectionMethod : "<<intersectionMethod->getClass()->className<<std::endl;
        }
    }
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::boxPrune(){
    //std::cout<<"boxPrune"<<std::endl;
    core::collision::NarrowPhaseDetection::beginNarrowPhase();

    _cur_axis = greatestVarianceAxis();

    int axis1 = (1  << _cur_axis) & 3;
    int axis2 = (1  << axis1) & 3;

    sofa::helper::AdvancedTimer::stepBegin("Incr SAP intersection");

    std::deque<int> active_boxes;//active boxes are the one that we encoutered only their min (end point), so if there are two boxes b0 and b1,
                                 //if we encounter b1_min as b0_min < b1_min, on the current axis, the two boxes intersect :  b0_min--------------------b0_max
                                 //                                                                                                      b1_min---------------------b1_max
                                 //once we encouter b0_max, b0 will not intersect with nothing (trivial), so we delete it from active_boxes.
                                 //so the rule is : -every time we encounter a box min end point, we check if it is overlapping with other active_boxes and add the owner (a box) of this end point to
                                 //                  the active boxes.
                                 //                 -every time we encounter a max end point of a box, we are sure that we encountered min end point of a box because _end_points is sorted,
                                 //                  so, we delete the owner box, of this max end point from the active boxes
    for(typename EndPointList::iterator it = _end_points[_cur_axis].begin() ; it != _end_points[_cur_axis].end() ; ++it){
        if((**it).max()){//erase it from the active_boxes
            assert(std::find(active_boxes.begin(),active_boxes.end(),(**it).boxID()) != active_boxes.end());
            active_boxes.erase(std::find(active_boxes.begin(),active_boxes.end(),(**it).boxID()));
        }
        else{//we encounter a min possible intersection between it and active_boxes
            int new_box = (**it).boxID();

            //SAPBox & box0 = _boxes[new_box];
            for(unsigned int i = 0 ; i < active_boxes.size() ; ++i){
                //SAPBox & box1 = _boxes[active_boxes[i]];

                addIfCollide(new_box,active_boxes[i],axis1,axis2);
            }
            active_boxes.push_back(new_box);
        }
    }

    //std::cout<<"boxPrune==============="<<std::endl;
    sofa::helper::AdvancedTimer::stepEnd("Incr SAP intersection");
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::removeCollision(int a,int b){
    core::CollisionModel *finalcm1 = _boxes[a].cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
    core::CollisionModel *finalcm2 = _boxes[b].cube.getCollisionModel()->getLast();

    bool swap;
    if((finalcm1->isSimulated() || finalcm2->isSimulated()) &&//check if the two boxes could be in collision, if it is not the case they are not added to _colliding_elems
            (((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2))) && (intersectionMethod->findIntersector(finalcm1,finalcm2,swap) != 0x0)){
        //std::cout<<"REMOVING !!"<<std::endl;
        _colliding_elems.remove(a,b);
    }
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::beginNarrowPhase(){
    this->NarrowPhaseDetection::beginNarrowPhase();
    updateEndPoints();

    if(_nothing_added){
        updateMovingBoxes();
    }
    else{
        reinitDetection();

        boxPrune();
    }

    //std::cout<<"number of collisions "<<_colliding_elems.size()<<std::endl;

    _colliding_elems.intersect(this);
    _nothing_added = true;
}

template <template<class T,class Allocator> class List,template <class T> class Allocator>
void TIncrSAP<List,Allocator>::updateMovingBoxes(){
    if(_boxes.size() < 2)
        return;

    //std::cout<<__LINE__<<std::endl;
    //std::cout<<"begin moving boxes"<<std::endl;
    EndPointID * cur_end_point;
    int axis1,axis2;
    typename EndPointList::iterator it,next_it,prev_it;

    //std::cout<<__LINE__<<std::endl;

    for(unsigned int i = 0 ; i < _boxes.size() ; ++i){
        //std::cout<<__LINE__<<std::endl;
        ISAPBox & cur_box = _boxes[i];
        if(_boxes[i].moving()){
            //std::cout<<"\tMOVED!"<<std::endl;
            for(int dim = 0 ; dim < 3 ; ++dim){
                //std::cout<<"axis number "<<dim<<std::endl;
                //std::cout<<"position "<<(_boxes[i].cube.minVect() + _boxes[i].cube.maxVect())/2.0<<std::endl;
                //std::cout<<__LINE__<<std::endl;
                //first handle moving of min
                cur_end_point = &(cur_box.min(dim));
                it = _end_points[dim].begin() + cur_end_point->ID;

                //std::cout<<__LINE__<<std::endl;

                next_it = it;
                ++next_it;

                //std::cout<<__LINE__<<std::endl;

                prev_it = it;
                if(it != _end_points[dim].begin())
                    --prev_it;

                //std::cout<<__LINE__<<std::endl;

                if((next_it != _end_points[dim].end()) && cur_end_point->value > (**next_it).value){
                    do{//moving the min forward
                        if((**next_it).max())//cur_min becomes greater than a max, so removing this contact if it exists
                            removeCollision(cur_end_point->boxID(),(**next_it).boxID());

                        //std::cout<<__LINE__<<std::endl;

                        --((**next_it).ID);
                        ++(cur_end_point->ID);
                        (*it) = (*next_it);
                        it = next_it;
                        ++next_it;
                    }
                    while((next_it != _end_points[dim].end()) && cur_end_point->value > (**next_it).value);

                    //std::cout<<__LINE__<<std::endl;

                    (*it) = cur_end_point;
                }
                else if(cur_end_point->value < (**prev_it).value){//moving the min backward
                    axis1 = (1  << dim) & 3;
                    axis2 = (1  << axis1) & 3;

                    do{
                        if((**prev_it).max())//min becomes inferior to a max => begginning of an intersection on this axis
                            addIfCollide(i,(**prev_it).boxID(),axis1,axis2);

                        //std::cout<<__LINE__<<std::endl;

                        ++((**prev_it).ID);
                        --(cur_end_point->ID);
                        (*it) = (*prev_it);
                        it = prev_it;

                        if(prev_it == _end_points[dim].begin())
                            break;

                        --prev_it;
//                        //std::cout<<"5.4"<<std::endl;
//                        //std::cout<<"THE ID "<<cur_end_point->ID<<std::endl;
//                        //std::cout<<"size "<<_end_points[dim].size()<<std::endl;
//                        if((**prev_it).value < 5){
//                            //std::cout<<"ha"<<std::endl;
//                        }
//                        //std::cout<<"5.5"<<std::endl;
                    }
                    while(cur_end_point->value < (**prev_it).value);

                    //std::cout<<__LINE__<<std::endl;

                    (*it) = cur_end_point;
                }

                //std::cout<<__LINE__<<std::endl;

                //handling moving of max
                cur_end_point = &(cur_box.max(dim));
                it = _end_points[dim].begin() + cur_end_point->ID;

                next_it = it;
                ++next_it;

                //std::cout<<__LINE__<<std::endl;
                //std::cout<<"cur_end_point->ID "<<cur_end_point->ID<<std::endl;
                //std::cout<<"(*next_it) "<<(*next_it)<<std::endl;

                prev_it = it;
                if(it != _end_points[dim].begin())
                    --prev_it;

                //std::cout<<"(*prev_it) "<<(*prev_it)<<std::endl;

                //std::cout<<__LINE__<<std::endl;

                if((next_it != _end_points[dim].end()) && cur_end_point->value > (**next_it).value){
                    axis1 = (1  << dim) & 3;
                    axis2 = (1  << axis1) & 3;
                    //std::cout<<__LINE__<<std::endl;

                    do{
                        if((**next_it).min())
                            addIfCollide(i,(**next_it).boxID(),axis1,axis2);

                        //std::cout<<__LINE__<<std::endl;

                        --((**next_it).ID);
                        ++(cur_end_point->ID);
                        (*it) = (*next_it);
                        it = next_it;
                        ++next_it;
                    }
                    while((next_it != _end_points[dim].end()) && cur_end_point->value > (**next_it).value);

                    (*it) = cur_end_point;
                }
                else if(cur_end_point->value < (**prev_it).value){//moving the max backward
                    do{
                        //std::cout<<__LINE__<<std::endl;

                        if((**prev_it).min())//max becomes inferior to a min => end of an intersection on this axis
                            removeCollision(cur_end_point->boxID(),(**prev_it).boxID());

                        //std::cout<<__LINE__<<std::endl;

                        ++((**prev_it).ID);
                        --(cur_end_point->ID);
                        (*it) = (*prev_it);
                        it = prev_it;

                        if(prev_it == _end_points[dim].begin())
                            break;

                        --prev_it;
                    }
                    while(cur_end_point->value < (**prev_it).value);

                    //std::cout<<__LINE__<<std::endl;

                    (*it) = cur_end_point;
                }
            }
        }
    }

    //std::cout<<"end updateMovingBoxes"<<std::endl;
}


}
}
}
#endif // INCRSAP_INL
