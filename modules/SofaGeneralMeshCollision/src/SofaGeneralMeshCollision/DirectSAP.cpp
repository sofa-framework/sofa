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
#include <SofaGeneralMeshCollision/DirectSAP.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.h>
#include <queue>


namespace sofa::component::collision
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

    if(min0[axis] > max1[axis])
    {
        return std::pow(min0[axis] - max1[axis], 2);
    }

    if(min1[axis] > max0[axis])
    {
        return std::pow((min1[axis] - max0[axis]), 2);
    }

    return 0;
}

DirectSAP::DirectSAP()
    : bDraw(initData(&bDraw, false, "draw", "enable/disable display of results"))
    , box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored"))
    , _cur_axis(0)
    , _alarmDist(0)
    , _alarmDist_d2(0)
    , _sq_alarmDist(0)
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
        if (!boxModel) boxModel = sofa::core::objectmodel::New<CubeCollisionModel>();
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

    if(_new_cm.empty())
        return;

    //to gain time, we create at the same time all SAPboxes so as to allocate
    //memory the less times
    std::vector<CubeCollisionModel*> cube_models;
    cube_models.reserve(_new_cm.size());

    int n = 0;
    for(unsigned int i = 0 ; i < _new_cm.size() ; ++i){
        n += _new_cm[i]->getSize();
        cube_models.push_back(dynamic_cast<CubeCollisionModel*>(_new_cm[i]->getPrevious()));
    }

    _boxes.reserve(_boxes.size() + n);
    EndPoint * end_pts = new EndPoint[2*n];
    _to_del.push_back(end_pts);

    int cur_EndPtID = 0;
    int cur_boxID = static_cast<int>(_boxes.size());
    for(unsigned int i = 0 ; i < cube_models.size() ; ++i){
        CubeCollisionModel * cm = cube_models[i];
        for(Size j = 0 ; j < cm->getSize() ; ++j){
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

int DirectSAP::greatestVarianceAxis() const
{
    defaulttype::Vector3 variance;//variances for each axis
    defaulttype::Vector3 mean;//means for each axis

    //computing the mean value of end points on each axis
    for (const auto& dsapBox : _boxes)
    {
        mean += dsapBox.cube.minVect();
        mean += dsapBox.cube.maxVect();
    }

    const auto nbBoxes = _boxes.size();
    if (nbBoxes > 0)
    {
        mean[0] /= 2. * static_cast<double>(nbBoxes);
        mean[1] /= 2. * static_cast<double>(nbBoxes);
        mean[2] /= 2. * static_cast<double>(nbBoxes);
    }

    //computing the variance of end points on each axis
    for (const auto& dsapBox : _boxes)
    {
        const defaulttype::Vector3 & min = dsapBox.cube.minVect();
        const defaulttype::Vector3 & max = dsapBox.cube.maxVect();

        for (unsigned int j = 0 ; j < 3; ++j)
        {
            variance[j] += std::pow(min[j] - mean[j], 2);
            variance[j] += std::pow(max[j] - mean[j], 2);
        }
    }

    if(variance[0] >= variance[1] && variance[0] >= variance[2])
        return 0;
    if(variance[1] >= variance[2])
        return 1;
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

    sofa::helper::AdvancedTimer::stepBegin("Direct SAP std::sort");
    std::sort(_end_points.begin(),_end_points.end(), CompPEndPoint());
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


                    if(finalintersector != nullptr){
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

inline void DSAPBox::show()const
{
    msg_info("DSAPBox") <<"MIN "<<cube.minVect()<< msgendl
                        <<"MAX "<<cube.maxVect() ;
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

int DirectSAPClass = core::RegisterObject("Collision detection using sweep and prune")
        .add< DirectSAP >()
        ;

} // namespace sofa::component::collision

