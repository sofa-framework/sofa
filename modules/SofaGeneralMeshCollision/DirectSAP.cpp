/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaGeneralMeshCollision/DirectSAP.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/Sphere.h>
#include <SofaMeshCollision/Triangle.h>
#include <SofaMeshCollision/Line.h>
#include <SofaMeshCollision/Point.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/ObjectFactory.h>
#include <map>
#include <queue>
#include <stack>

#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace component
{

namespace collision
{



inline void DSAPBox::update(int axis, double alarmDist){
    min->value = (cube.minVect())[axis] - alarmDist;
    max->value = (cube.maxVect())[axis] + alarmDist;
}


inline double DSAPBox::squaredDistance(const DSAPBox & other,int axis)const{
    const defaulttype::Vector3 & min0 = this->cube.minVect();
    const defaulttype::Vector3 & max0 = this->cube.maxVect();
    const defaulttype::Vector3 & min1 = other.cube.minVect();
    const defaulttype::Vector3 & max1 = other.cube.maxVect();

    double temp;

    if(min0[axis] > max1[axis]){
        temp = (min0[axis] - max1[axis]);
        return temp * temp;
    }
    else if(min1[axis] > max0[axis]){
        temp = (min1[axis] - max0[axis]);
        return temp * temp;
    }

    return 0;
}


inline bool DSAPBox::overlaps(const DSAPBox &other, int axis, double alarmDist) const{
    const defaulttype::Vector3 & min0 = this->cube.minVect();
    const defaulttype::Vector3 & max0 = this->cube.maxVect();
    const defaulttype::Vector3 & min1 = other.cube.minVect();
    const defaulttype::Vector3 & max1 = other.cube.maxVect();

    if(min0[axis] >= max1[axis] + alarmDist || min1[axis] >= max0[axis] + alarmDist)
        return false;

    return true;
}


DirectSAP::DirectSAP()
    : bDraw(initData(&bDraw, false, "draw", "enable/disable display of results"))
    , box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored"))
{
}


DirectSAP::~DirectSAP()
{

    for(unsigned int i = 0 ; i < _to_del.size() ; ++i)
        delete[] _to_del[i];
}


void DirectSAP::init()
{
    reinit();
}


void DirectSAP::reinit()
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
}


inline bool DirectSAP::added(core::CollisionModel *cm) const
{
    return collisionModels.count(cm->getLast()) >= 1;
}

inline void DirectSAP::add(core::CollisionModel *cm)
{
    collisionModels.insert(cm->getLast());
    _new_cm.push_back(cm->getLast());
}



void DirectSAP::endBroadPhase()
{
    BroadPhaseDetection::endBroadPhase();

    if(_new_cm.size() == 0)
        return;

    //to gain time, we create at the same time all SAPboxes so as to allocate
    //memory the less times
    std::vector<CubeModel*> cube_models;
    cube_models.reserve(_new_cm.size());

    int n = 0;
    for(unsigned int i = 0 ; i < _new_cm.size() ; ++i){
        n += _new_cm[i]->getSize();
        cube_models.push_back(dynamic_cast<CubeModel*>(_new_cm[i]->getPrevious()));
    }

    _boxes.reserve(_boxes.size() + n);
    EndPoint * end_pts = new EndPoint[2*n];
    _to_del.push_back(end_pts);

    int cur_EndPtID = 0;
    int cur_boxID = _boxes.size();
    for(unsigned int i = 0 ; i < cube_models.size() ; ++i){
        CubeModel * cm = cube_models[i];
        for(int j = 0 ; j < cm->getSize() ; ++j){
            EndPoint * min = &end_pts[cur_EndPtID];
            ++cur_EndPtID;
            EndPoint * max = &end_pts[cur_EndPtID];
            ++cur_EndPtID;

            min->setBoxID(cur_boxID);
            max->setBoxID(cur_boxID);
            max->setMax();

            _end_points.push_back(min);
            _end_points.push_back(max);

            _boxes.push_back(DSAPBox(Cube(cm,j),min,max));
            ++cur_boxID;
        }
    }

    _new_cm.clear();
}



void DirectSAP::addCollisionModel(core::CollisionModel *cm){
    if(!added(cm))
        add(cm);
}

int DirectSAP::greatestVarianceAxis()const{
    double diff;
    double v[3];//variances for each axis
    double m[3];//means for each axis
    for(int i = 0 ; i < 3 ; ++i)
        v[i] = m[i] = 0;

    //computing the mean value of end points on each axis
    for(unsigned int i = 0 ; i < _boxes.size() ; ++i){
        const defaulttype::Vector3 & min = _boxes[i].cube.minVect();
        const defaulttype::Vector3 & max = _boxes[i].cube.maxVect();
        m[0] += min[0] + max[0];
        m[1] += min[1] + max[1];
        m[2] += min[2] + max[2];
    }

    m[0] /= 2*_boxes.size();
    m[1] /= 2*_boxes.size();
    m[2] /= 2*_boxes.size();

    //computing the variance of end points on each axis
    for(unsigned int i = 0 ; i < _boxes.size() ; ++i){
        const defaulttype::Vector3 & min = _boxes[i].cube.minVect();
        const defaulttype::Vector3 & max = _boxes[i].cube.maxVect();

        diff = min[0] - m[0];
        v[0] += diff*diff;
        diff = max[0] - m[0];
        v[0] += diff*diff;

        diff = min[1] - m[1];
        v[1] += diff*diff;
        diff = max[1] - m[1];
        v[1] += diff*diff;

        diff = min[2] - m[2];
        v[2] += diff*diff;
        diff = max[2] - m[2];
        v[2] += diff*diff;
    }

    if(v[0] >= v[1] && v[0] >= v[2])
        return 0;
    else if(v[1] >= v[2])
        return 1;
    else
        return 2;
}



void DirectSAP::update(){
    _cur_axis = greatestVarianceAxis();
    for(unsigned int i = 0 ; i < _boxes.size() ; ++i){
        _boxes[i].update(_cur_axis,_alarmDist_d2);
    }
}

void DirectSAP::beginNarrowPhase()
{
    core::collision::NarrowPhaseDetection::beginNarrowPhase();
    _alarmDist = getIntersectionMethod()->getAlarmDistance();
    _sq_alarmDist = _alarmDist * _alarmDist;
    _alarmDist_d2 = _alarmDist/2.0;

    update();

    CompPEndPoint comp;

    sofa::helper::AdvancedTimer::stepBegin("Direct SAP std::sort");
    std::sort(_end_points.begin(),_end_points.end(),comp);
    sofa::helper::AdvancedTimer::stepEnd("Direct SAP std::sort");

    sofa::helper::AdvancedTimer::stepBegin("Direct SAP intersection");

    std::deque<int> active_boxes;//active boxes are the one that we encoutered only their min (end point), so if there are two boxes b0 and b1,
                                 //if we encounter b1_min as b0_min < b1_min, on the current axis, the two boxes intersect :  b0_min--------------------b0_max
                                 //                                                                                                      b1_min---------------------b1_max
                                 //once we encouter b0_max, b0 will not intersect with nothing (trivial), so we delete it from active_boxes.
                                 //so the rule is : -every time we encounter a box min end point, we check if it is overlapping with other active_boxes and add the owner (a box) of this end point to
                                 //                  the active boxes.
                                 //                 -every time we encounter a max end point of a box, we are sure that we encountered min end point of a box because _end_points is sorted,
                                 //                  so, we delete the owner box, of this max end point from the active boxes
    for(EndPointList::iterator it = _end_points.begin() ; it != _end_points.end() ; ++it){
        if((**it).max()){//erase it from the active_boxes
            assert(std::find(active_boxes.begin(),active_boxes.end(),(**it).boxID()) != active_boxes.end());
            active_boxes.erase(std::find(active_boxes.begin(),active_boxes.end(),(**it).boxID()));
        }
        else{//we encounter a min possible intersection between it and active_boxes
            int new_box = (**it).boxID();

            DSAPBox & box0 = _boxes[new_box];
            for(unsigned int i = 0 ; i < active_boxes.size() ; ++i){
                DSAPBox & box1 = _boxes[active_boxes[i]];

                core::CollisionModel *finalcm1 = box0.cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
                core::CollisionModel *finalcm2 = box1.cube.getCollisionModel()->getLast();
                if((finalcm1->isSimulated() || finalcm2->isSimulated()) &&
                        (((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2)) &&
                         /*box0.overlaps(box1,axis1,_alarmDist) && box0.overlaps(box1,axis2,_alarmDist)*/
                         box0.squaredDistance(box1) <= _sq_alarmDist)){//intersection on all axes

                    bool swapModels = false;
                    core::collision::ElementIntersector* finalintersector = intersectionMethod->findIntersector(finalcm1, finalcm2, swapModels);//find the method for the finnest CollisionModels

                    assert(box0.cube.getExternalChildren().first.getIndex() == box0.cube.getIndex());
                    assert(box1.cube.getExternalChildren().first.getIndex() == box1.cube.getIndex());

                    if((!swapModels) && finalcm1->getClass() == finalcm2->getClass() && finalcm1 > finalcm2)//we do that to have only pair (p1,p2) without having (p2,p1)
                        swapModels = true;


                    if(finalintersector != 0x0){
                        if(swapModels){
                            sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(finalcm2, finalcm1);
                            finalintersector->beginIntersect(finalcm2, finalcm1, outputs);//creates outputs if null

                            finalintersector->intersect(box1.cube.getExternalChildren().first,box0.cube.getExternalChildren().first,outputs) ;
                        }
                        else{
                            sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(finalcm1, finalcm2);

                            finalintersector->beginIntersect(finalcm1, finalcm2, outputs);//creates outputs if null

                            finalintersector->intersect(box0.cube.getExternalChildren().first,box1.cube.getExternalChildren().first,outputs) ;
                        }
                    }
                    else{
                    }
                }
            }
            active_boxes.push_back(new_box);
        }
    }
    sofa::helper::AdvancedTimer::stepEnd("Direct SAP intersection");
}

bool DSAPBox::overlaps(const DSAPBox &other,double alarmDist) const{
    return overlaps(other,0,alarmDist) && overlaps(other,0,alarmDist) && overlaps(other,0,alarmDist);
}

double DSAPBox::squaredDistance(const DSAPBox & other)const{
    double dist2 = 0;

    for(int axis = 0 ; axis < 3 ; ++axis){
        dist2 += squaredDistance(other,axis);
    }

    return dist2;
}

using namespace sofa::defaulttype;
using namespace collision;

SOFA_DECL_CLASS(DirectSap)


int DirectSAPClass = core::RegisterObject("Collision detection using sweep and prune")
        .add< DirectSAP >()
        ;


} // namespace collision

} // namespace component

} // namespace sofa

