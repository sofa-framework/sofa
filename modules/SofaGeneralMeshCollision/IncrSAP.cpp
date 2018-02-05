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
#include <sofa/core/ObjectFactory.h>
#include <SofaGeneralMeshCollision/IncrSAP.h>

namespace sofa
{

namespace component
{

namespace collision
{


/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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


inline EndPointID & ISAPBox::min(int dim){return *(_min[dim]);}
inline const EndPointID & ISAPBox::min(int dim)const{return *(_min[dim]);}

inline EndPointID & ISAPBox::max(int dim){return *(_max[dim]);}
inline const EndPointID & ISAPBox::max(int dim)const{return *(_max[dim]);}

inline double ISAPBox::curMin(int dim)const{return cube.minVect()[dim];}
inline double ISAPBox::curMax(int dim)const{return cube.maxVect()[dim];}

inline void ISAPBox::updatedMin(int dim,EndPointID & end_point, double alarmDist)const{
    end_point = (*_min[dim]);
    end_point.value = cube.minVect()[dim] - alarmDist;
}

inline void ISAPBox::updatedMax(int dim,EndPointID & end_point,double alarmDist)const{
    end_point = (*_max[dim]);
    end_point.value = cube.maxVect()[dim] + alarmDist;
}

inline void ISAPBox::update(double alarmDist){
    for(int i = 0 ; i < 3 ; ++i){
        _min[i]->value = cube.minVect()[i] - alarmDist;
        _max[i]->value = cube.maxVect()[i] + alarmDist;
    }
}

inline void ISAPBox::updateMin(int dim,double alarmDist){
    _min[dim]->value = cube.minVect()[dim] - alarmDist;
}

inline void ISAPBox::updateMax(int dim,double alarmDist){
    _max[dim]->value = cube.maxVect()[dim] + alarmDist;
}

inline bool ISAPBox::endPointsAreAlright(int ID){
    for(int i = 0 ; i < 3 ; ++i){
        if(!_min[i]->min())
            return false;

        if(!_max[i]->max())
            return false;

        if(_min[i]->boxID() != ID)
            return false;

        if(_max[i]->boxID() != ID)
            return false;
    }

    return true;
}


inline void ISAPBox::init(int boxID,EndPointID ** endPts){
    for(int i = 0 ; i < 3 ; ++i){
        _min[i] = endPts[i];
        _max[i] = endPts[3 + i];
    }

    for(int i = 0 ; i < 3 ; ++i){
        _min[i]->setMinAndBoxID(boxID);
        _max[i]->setMaxAndBoxID(boxID);
//        _min[i]->setBoxID(boxID);
//        _max[i]->setBoxID(boxID);
//        _min[i]->setMin();
//        _max[i]->setMax();
    }

    //update();
}

inline bool ISAPBox::endPointsOverlap(const ISAPBox & other, int axis) const{
    assert(axis >= 0);
    assert(axis < 3);
//    const Vector3 & minVect_this = cube.minVect();
//    const Vector3 & maxVect_this = cube.maxVect();
//    const Vector3 & minVect_other = other.cube.minVect();
//    const Vector3 & maxVect_other = other.cube.maxVect();


    if((min(axis).value >= other.max(axis).value) || (other.min(axis).value >= max(axis).value))
        return false;

    return true;
}

inline bool ISAPBox::minMoving(int axis,double alarmDist) const{
    return min(axis).value  != cube.minVect()[axis] - alarmDist;
}

inline bool ISAPBox::maxMoving(int axis,double alarmDist) const{
    return max(axis).value != cube.maxVect()[axis] + alarmDist;
}

inline bool ISAPBox::moving(int axis,double alarmDist) const{
    return minMoving(axis,alarmDist) || maxMoving(axis,alarmDist);
//    const core::CollisionElementIterator & finE = finalElement();

//    const core::CollisionModel * cm = finE.getCollisionModel();

//    switch (cm->getEnumType()){
//        case core::CollisionModel::OBB_TYPE:
//            return fabs(((static_cast<const OBBModel*>(cm))->lvelocity(finE.getIndex()))[axis]) > tolerance;
//            break;
//        case core::CollisionModel::CAPSULE_TYPE:
//            return fabs(((static_cast<const CapsuleModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
//            break;
//        case core::CollisionModel::SPHERE_TYPE:
//            return fabs(((static_cast<const SphereModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
//            break;
//        case core::CollisionModel::TRIANGLE_TYPE:
//            return fabs(((static_cast<const TriangleModel*>(cm))-> velocity(finE.getIndex()))[axis]) > tolerance;
//            break;
//        case core::CollisionModel::LINE_TYPE:
//            return fabs(((static_cast<const LineModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
//            break;
//        case core::CollisionModel::POINT_TYPE:
//            return fabs(((static_cast<const PointModel*>(cm))->velocity(finE.getIndex()))[axis]) > tolerance;
//            break;
//        default:
//            msg_info()<<"CollisionModel type not found within SAPBox::moving"<<std::endl;
//            return true;
//    }
}

inline bool ISAPBox::moving(double alarmDist) const{
    return moving(0,alarmDist) || moving(1,alarmDist) || moving(2,alarmDist);
//    const core::CollisionElementIterator & finE = finalElement();

//    const core::CollisionModel * cm = finE.getCollisionModel();

//    switch (cm->getEnumType()){
//        case core::CollisionModel::OBB_TYPE:
//            return ((static_cast<const OBBModel*>(cm))->lvelocity(finE.getIndex())).norm2() > tolerance*tolerance;
//            break;
//        case core::CollisionModel::CAPSULE_TYPE:
//            return ((static_cast<const CapsuleModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
//            break;
//        case core::CollisionModel::SPHERE_TYPE:
//            return ((static_cast<const SphereModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
//            break;
//        case core::CollisionModel::TRIANGLE_TYPE:
//            return ((static_cast<const TriangleModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
//            break;
//        case core::CollisionModel::LINE_TYPE:
//            return ((static_cast<const LineModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
//            break;
//        case core::CollisionModel::POINT_TYPE:
//            return ((static_cast<const PointModel*>(cm))->velocity(finE.getIndex())).norm2() > tolerance*tolerance;
//            break;
//        default:
//            msg_info()<<"CollisionModel type not found within SAPBox::moving"<<std::endl;
//            return true;
//    }
}

inline const core::CollisionElementIterator ISAPBox::finalElement()const{
    return cube.getExternalChildren().first;
}


IncrSAP::IncrSAP()
    : bDraw(initData(&bDraw, false, "draw", "enable/disable display of results"))
    , box(initData(&box, "box", "if not empty, objects that do not intersect this bounding-box will be ignored")),
      _nothing_added(true)
{
    //_end_points = new EndPointList[3];
}


IncrSAP::~IncrSAP(){
    for(int i = 0 ; i < 3 ; ++i)
        for(EndPointList::iterator it = _end_points[i].begin() ; it != _end_points[i].end() ; ++it)
            delete (*it);


    //delete[] _end_points;
}



void IncrSAP::purge(){
    for(int i = 0 ; i < 3 ; ++i){
        for(EndPointList::iterator it = _end_points[i].begin() ; it != _end_points[i].end() ; ++it)
            delete (*it);

        _end_points[i].clear();
    }

    _boxes.clear();
    _colliding_elems.clear();
    collisionModels.clear();
}


void IncrSAP::init()
{
    reinit();
}

//
//void IncrSAP::initIntersectors(){
//    for(typename std::set<CollModID>::const_iterator it = collisionModels.begin() ; it != collisionModels.end() ; ++it){
//        for(std::set<CollModID>::const_iterator it2 = it ; it2 != collisionModels.end() ; ++it2){

//        }
//    }
//}


void IncrSAP::reinit()
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


inline bool IncrSAP::added(core::CollisionModel *cm) const
{
    return collisionModels.count(cm->getLast()) >= 1;
}


inline bool IncrSAP::add(core::CollisionModel *cm)
{
    return (collisionModels.insert(cm->getLast())).second;
}


inline void IncrSAP::addCollisionModel(core::CollisionModel *cm)
{
    if(add(cm)){
        _colliding_elems.add(cm->getLast(),intersectionMethod);
        _nothing_added = false;

        CubeModel * cube_model = dynamic_cast<CubeModel *>(cm->getLast()->getPrevious());
        assert(cube_model->getPrevious() == cm->getFirst());

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

            assert(new_box.endPointsAreAlright(i + old_size));
        }
    }
}


int IncrSAP::greatestVarianceAxis()const{
    double diff;
    double v[3];//variances for each axis
    double m[3];//means for each axis
    for(int i = 0 ; i < 3 ; ++i)
        v[i] = m[i] = 0;

    //computing the mean value of end points on each axis
    for(int j = 0 ; j < 3 ; ++j)
        for(EndPointList::const_iterator it = _end_points[j].begin() ; it != _end_points[j].end() ; ++it)
            m[j] += (**it).value;

    m[0] /= 2*_boxes.size();
    m[1] /= 2*_boxes.size();
    m[2] /= 2*_boxes.size();

    //computing the variance of end points on each axis
    for(int j = 0 ; j < 3 ; ++j){
        for(EndPointList::const_iterator it = _end_points[j].begin() ; it != _end_points[j].end() ; ++it){
            diff = (**it).value - m[j];
            v[j] += diff*diff;
        }
    }

    if(v[0] >= v[1] && v[0] >= v[2])
        return 0;
    else if(v[1] >= v[2])
        return 1;
    else
        return 2;
}


void IncrSAP::updateEndPoints(){
    for(unsigned int i = 0 ; i < _boxes.size() ; ++i){
        _boxes[i].update(_alarmDist_d2);
        assert(_boxes[i].endPointsAreAlright(i));
    }
}


void IncrSAP::setEndPointsID(){
    for(int dim = 0 ; dim < 3 ; ++dim){
        int ID = 0;
        for(EndPointList::iterator it = _end_points[dim].begin() ; it != _end_points[dim].end() ; ++it){
            (**it).ID = ID;
            ++ID;
        }
    }
}


void IncrSAP::reinitDetection(){
    _colliding_elems.clear();
    CompPEndPoint comp;
    for(int j = 0 ; j < 3 ; ++j){
        std::sort(_end_points[j].begin(),_end_points[j].end(),comp);
    }
    setEndPointsID();
}



void IncrSAP::showEndPoints()const{
    for(int j = 0 ; j < 3 ; ++j){
        msg_info() <<"dimension "<<j<<"===========" ;
        for(EndPointList::const_iterator it = _end_points[j].begin() ; it != _end_points[j].end() ; ++it){
            const EndPointID & end_pt = (**it);
            end_pt.show();
        }
    }
}


void IncrSAP::showBoxes()const{
    for(size_t i = 0 ; i < _boxes.size() ; ++i){
        const ISAPBox & box = _boxes[i];
        std::stringstream tmp;

        tmp <<"collision model "<<box.cube.getCollisionModel()->getLast()<<" index "<<box.cube.getExternalChildren().first.getIndex()<<msgendl ;

        tmp<<"minBBox ";
        for(int j = 0 ; j < 3 ; ++j){
            tmp<<" "<<box.min(j).value;
        }
        tmp<<msgendl ;

        tmp<<"maxBBox ";
        for(int j = 0 ; j < 3 ; ++j){
            tmp<<" "<<box.max(j).value;
        }
        msg_info() << tmp.str() ;
    }
}


void IncrSAP::addIfCollide(int boxID1,int boxID2){
    if(boxID1 == boxID2)
        return;

    assert(boxID1 < (int)(_boxes.size()));
    assert(boxID2 < (int)(_boxes.size()));

    ISAPBox & box0 = _boxes[boxID1];
    ISAPBox & box1 = _boxes[boxID2];
    core::CollisionModel *finalcm1 = box0.cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
    core::CollisionModel *finalcm2 = box1.cube.getCollisionModel()->getLast();

    if((finalcm1->isSimulated() || finalcm2->isSimulated()) &&
            (((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2)) && box0.overlaps(box1,_alarmDist))){//intersection on all axes

         _colliding_elems.add(boxID1,boxID2,box0.finalElement(),box1.finalElement());
    }
}



void IncrSAP::addIfCollide(int boxID1,int boxID2,int axis1,int axis2){
    if(boxID1 == boxID2)
        return;

    assert(boxID1 < (int)(_boxes.size()));
    assert(boxID2 < (int)(_boxes.size()));

    ISAPBox & box0 = _boxes[boxID1];
    ISAPBox & box1 = _boxes[boxID2];
    core::CollisionModel *finalcm1 = box0.cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
    core::CollisionModel *finalcm2 = box1.cube.getCollisionModel()->getLast();

    if((finalcm1->isSimulated() || finalcm2->isSimulated()) &&
            (((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2)) && box0.endPointsOverlap(box1,axis1) && box0.endPointsOverlap(box1,axis2))){//intersection on all axes

                _colliding_elems.add(boxID1,boxID2,box0.finalElement(),box1.finalElement());
    }
}



void IncrSAP::boxPrune(){
    _cur_axis = greatestVarianceAxis();
    int axis1 = (1  << _cur_axis) & 3;
    int axis2 = (1  << axis1) & 3;

    sofa::helper::AdvancedTimer::stepBegin("Box Prune SAP intersection");

    std::deque<int> active_boxes;//active boxes are the one that we encoutered only their min (end point), so if there are two boxes b0 and b1,
                                 //if we encounter b1_min as b0_min < b1_min, on the current axis, the two boxes intersect :  b0_min--------------------b0_max
                                 //                                                                                                      b1_min---------------------b1_max
                                 //once we encouter b0_max, b0 will not intersect with nothing (trivial), so we delete it from active_boxes.
                                 //so the rule is : -every time we encounter a box min end point, we check if it is overlapping with other active_boxes and add the owner (a box) of this end point to
                                 //                  the active boxes.
                                 //                 -every time we encounter a max end point of a box, we are sure that we encountered min end point of a box because _end_points is sorted,
                                 //                  so, we delete the owner box, of this max end point from the active boxes
    for(EndPointList::iterator it = _end_points[_cur_axis].begin() ; it != _end_points[_cur_axis].end() ; ++it){
        if((**it).max()){//erase it from the active_boxes
            assert(std::find(active_boxes.begin(),active_boxes.end(),(**it).boxID()) != active_boxes.end());
            active_boxes.erase(std::find(active_boxes.begin(),active_boxes.end(),(**it).boxID()));
        }
        else{//we encounter a min possible intersection between it and active_boxes
            int new_box = (**it).boxID();

            for(unsigned int i = 0 ; i < active_boxes.size() ; ++i){

                addIfCollide(new_box,active_boxes[i],axis1,axis2);
            }
            active_boxes.push_back(new_box);
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("Box Prune SAP intersection");
}


void IncrSAP::removeCollision(int a,int b){
    if(a == b)
        return;

    core::CollisionModel *finalcm1 = _boxes[a].cube.getCollisionModel()->getLast();//get the finnest CollisionModel which is not a CubeModel
    core::CollisionModel *finalcm2 = _boxes[b].cube.getCollisionModel()->getLast();

    bool swap;
    if((!(_boxes[a].overlaps(_boxes[b],_alarmDist))) && //check if it really doesn't overlap
            (finalcm1->isSimulated() || finalcm2->isSimulated()) &&//check if the two boxes could be in collision, if it is not the case they are not added to _colliding_elems
            (((finalcm1->getContext() != finalcm2->getContext()) || finalcm1->canCollideWith(finalcm2))) && (intersectionMethod->findIntersector(finalcm1,finalcm2,swap) != 0x0)){
        _colliding_elems.remove(a,b,_boxes[a].finalElement(),_boxes[b].finalElement());
    }
}


void IncrSAP::beginNarrowPhase(){
    this->NarrowPhaseDetection::beginNarrowPhase();
    _alarmDist = getIntersectionMethod()->getAlarmDistance();
    _alarmDist_d2 = _alarmDist/2.0;

    if(_nothing_added){
        updateMovingBoxes();
    }
    else{
        updateEndPoints();
        reinitDetection();
        assert(assertion_end_points_sorted());
        boxPrune();
        assert(assertion_end_points_sorted());
    }

    _colliding_elems.intersect(this);
    assert(assertion_end_points_sorted());
    _nothing_added = true;
}


bool IncrSAP::assertion_order(EndPointList::iterator it,EndPointList::iterator begin,EndPointList::iterator end){
    CompPEndPoint comp;
    EndPointList::iterator next_it = it;++next_it;
    if(next_it != end && comp(*next_it,*it))
        return false;

    if(it != begin){
        EndPointList::iterator prev_it = it;--prev_it;
        if(comp(*it,*prev_it))
            return false;
    }

    return true;
}



bool IncrSAP::assertion_list_order(EndPointList::iterator begin_it,const EndPointList::iterator & end_it){
    CompPEndPoint inferior;
    EndPointList::iterator next_it = begin_it;
    ++next_it;
    for(;next_it != end_it ; ++next_it,++begin_it){
        if(inferior(*next_it,*begin_it))
            return false;
    }

    return true;
}



bool IncrSAP::assertion_superior(EndPointList::iterator begin_it,const EndPointList::iterator & end_it,EndPoint* point){
    CompPEndPoint inferior;
    for(;begin_it != end_it ;++begin_it){
        if(inferior(point,*begin_it)){
            inferior(point,*begin_it);
            inferior(*begin_it,point);
            return false;
        }
    }

    return true;
}


bool IncrSAP::assertion_inferior(EndPointList::iterator begin_it,const EndPointList::iterator & end_it,EndPoint* point){
    CompPEndPoint inferior;
    for(;begin_it != end_it ;++begin_it){
        if(inferior(*begin_it,point))
            return false;
    }

    return true;
}



bool IncrSAP::assertion_end_points_sorted() const{
    CompPEndPoint inferior;
    int n = 0;
    for(int dim = 0 ; dim < 3 ; ++dim){
        int ID = 0;
        EndPointList::const_iterator next_it2;
        int equality_number = 0;
        for(EndPointList::const_iterator it2 = _end_points[dim].begin() ; it2 != _end_points[dim].end() ; ++it2){
            assert((**it2).ID == ID);

            next_it2 = it2;
            ++next_it2;
            if(next_it2 != _end_points[dim].end()){
                assert((**next_it2).ID == ID + 1);

                if(!inferior(*it2,*next_it2)){
                    ++n;

                    if((**it2).value == (**next_it2).value)
                        ++equality_number;
                }
            }

            ++ID;
        }

        msg_info_when(n!=0)
                << "STOP !";

    }

    return n == 0;
}

void IncrSAP::moveMinForward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & next_it){
    CompPEndPoint inferior;
    do{
        if((**next_it).max())
            removeCollision(cur_end_point->boxID(),(**next_it).boxID());

        ++(cur_end_point->ID);
        --((**next_it).ID);
        (*it) = (*next_it);
        it = next_it;
        ++next_it;
    }
    while((next_it != _end_points[dim].end()) && (inferior(*next_it,cur_end_point)));

    (*it) = cur_end_point;
}



void IncrSAP::moveMaxForward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & next_it){
    CompPEndPoint inferior;
    do{
        if((**next_it).min())
            addIfCollide(cur_end_point->boxID(),(**next_it).boxID());

        ++(cur_end_point->ID);
        --((**next_it).ID);
        (*it) = (*next_it);
        it = next_it;
        ++next_it;
        }
    while((next_it != _end_points[dim].end()) && (inferior(*next_it,cur_end_point)));

    (*it) = cur_end_point;
}


void IncrSAP::moveMinBackward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & prev_it){
    CompPEndPoint inferior;
    do{
        if((**prev_it).max())
            addIfCollide(cur_end_point->boxID(),(**prev_it).boxID());

        ++((**prev_it).ID);
        --(cur_end_point->ID);
        (*it) = (*prev_it);
        it = prev_it;

        if(prev_it == _end_points[dim].begin())
            break;

        --prev_it;
    }
    while(inferior(cur_end_point,*prev_it));

    (*it) = cur_end_point;
}


void IncrSAP::moveMaxBackward(int dim,EndPointID * cur_end_point,EndPointList::iterator & it,EndPointList::iterator & prev_it){
    CompPEndPoint inferior;
    do{
        if((**prev_it).min())
            removeCollision(cur_end_point->boxID(),(**prev_it).boxID());

        ++((**prev_it).ID);
        --(cur_end_point->ID);
        (*it) = (*prev_it);
        it = prev_it;

        if(prev_it == _end_points[dim].begin())
            break;

        --prev_it;
    }
    while(inferior(cur_end_point,*prev_it));

    (*it) = cur_end_point;
}



void IncrSAP::updateMovingBoxes(){
    assert(assertion_end_points_sorted());
    CompPEndPoint inferior;

    if(_boxes.size() < 2)
        return;

    EndPointID * cur_end_point_min,*cur_end_point_max;
    cur_end_point_min = cur_end_point_max = 0x0;

    EndPointList::iterator it_min,next_it_min,prev_it_min,base_it_min,it_max,next_it_max,prev_it_max,base_it_max;
    bool min_updated,max_updated,min_moving,max_moving;
    EndPointID updated_min;
    EndPointID updated_max;

    for(unsigned int i = 0 ; i < _boxes.size() ; ++i){
        ISAPBox & cur_box = _boxes[i];
        for(int dim = 0 ; dim < 3 ; ++dim){
            min_updated = false;
            max_updated = false;

            //FIRST CREATING CONTACTS THEN DELETING, this order is very important, it doesn't work in the other sens
            //MOVING MAX FOREWARD
            if((max_moving = cur_box.maxMoving(dim,_alarmDist_d2))){
                cur_box.updatedMax(dim,updated_max,_alarmDist_d2);//we don't update directly update the max of the box but a copy of it, because when
                                                    //moving an end point, only one end point can change its value. In this case, we could
                                                    //update the value of the max but not move it, it would mean that the max could not be at its right place and when moving
                                                    //the min backward (below), the list would not be sorted...
                cur_end_point_max = &(cur_box.max(dim));
                it_max = _end_points[dim].begin() + cur_end_point_max->ID;
                base_it_max = it_max;
                assert((**it_max).ID == cur_end_point_max->ID);

                next_it_max = it_max;
                ++next_it_max;

                prev_it_max = it_max;
                if(it_max != _end_points[dim].begin())
                    --prev_it_max;

                if(next_it_max != _end_points[dim].end() && inferior(*next_it_max,&updated_max)){//moving the max foreward
                    cur_end_point_max->value = updated_max.value;//the real update of the end point (belonging to the end point list) is done
                                                                 //here because this end point will be put at its right place
                    moveMaxForward(dim,cur_end_point_max,it_max,next_it_max);
                    max_updated = true;
                }//after, cases when the end point is at its right place
                else if(next_it_max == _end_points[dim].end() && inferior(*prev_it_max,&updated_max)){
                    cur_end_point_max->value = updated_max.value;
                    max_updated = true;
                }
                else if(it_max == _end_points[dim].begin() && inferior(&updated_max,*next_it_max)){
                    cur_end_point_max->value = updated_max.value;
                    max_updated = true;
                }
                else if(inferior(*prev_it_max,&updated_max) && inferior(&updated_max,*next_it_max)){
                    cur_end_point_max->value = updated_max.value;
                    max_updated = true;
                }
            }

            //MOVING MIN BACKWARD
            if((min_moving = cur_box.minMoving(dim,_alarmDist_d2))){
                cur_box.updatedMin(dim,updated_min,_alarmDist_d2);
                cur_end_point_min = &(cur_box.min(dim));
                it_min = _end_points[dim].begin() + cur_end_point_min->ID;
                base_it_min = it_min;
                assert((**it_min).ID == cur_end_point_min->ID);

                next_it_min = it_min;
                ++next_it_min;

                prev_it_min = it_min;
                if(it_min != _end_points[dim].begin())
                    --prev_it_min;

                if((it_min != _end_points[dim].begin()) && inferior(&updated_min,*prev_it_min)){//moving the min backward
                    cur_end_point_min->value = updated_min.value;
                    moveMinBackward(dim,cur_end_point_min,it_min,prev_it_min);
                    min_updated = true;
                }//after, cases when the end point is at its right place
                else if(it_min == _end_points[dim].begin() && inferior(&updated_min,*next_it_min)){
                    cur_end_point_min->value = updated_min.value;
                    min_updated = true;
                }
                else if(next_it_min == _end_points[dim].end() && inferior(*prev_it_min,&updated_min)){
                    cur_end_point_min->value = updated_min.value;
                    min_updated = true;
                }
                else if(inferior(&updated_min,*next_it_min) && inferior(*prev_it_min,&updated_min)){
                    cur_end_point_min->value = updated_min.value;
                    min_updated = true;
                }
            }

            //THEN DELETING
            if(min_moving && (!min_updated)){
                cur_end_point_min->value = updated_min.value;

                //MOVING MIN FOREWARD
                if((next_it_min != _end_points[dim].end()) && (inferior(*next_it_min,cur_end_point_min))){
                    moveMinForward(dim,cur_end_point_min,it_min,next_it_min);
                }
            }

            //MOVING MAX BACKWARD
            if(max_moving && (!max_updated)){
                cur_end_point_max->value = updated_max.value;

                it_max = _end_points[dim].begin() + cur_end_point_max->ID;
                prev_it_max = it_max;
                if(it_max != _end_points[dim].begin())
                    --prev_it_max;


                if((prev_it_max != it_max && inferior(cur_end_point_max,*prev_it_max))){
                    moveMaxBackward(dim,cur_end_point_max,it_max,prev_it_max);
                }
            }

            if(min_moving || max_moving){
                assert(assertion_end_points_sorted());
            }
        }
    }
}




double ISAPBox::tolerance = (double)(1e-7);

double ISAPBox::squaredDistance(const ISAPBox & other) const{
    const defaulttype::Vector3 & min_vect0 = cube.minVect();
    const defaulttype::Vector3 & max_vect0 = cube.maxVect();
    const defaulttype::Vector3 & min_vect1 = other.cube.minVect();
    const defaulttype::Vector3 & max_vect1 = other.cube.maxVect();

    double temp;
    double dist2 = 0;

    for(int i = 0 ; i < 3 ; ++i){
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        if(max_vect0[i] <= min_vect1[i]){
            temp = max_vect0[i] - min_vect1[i];
            dist2 += temp * temp;
        }
        else if(max_vect1[i] <= min_vect0[i]){
            temp = max_vect1[i] - min_vect0[i];
            dist2 += temp * temp;
        }
    }

    return dist2;
}

bool ISAPBox::overlaps(const ISAPBox & other,double alarmDist) const{
    const defaulttype::Vector3 & min_vect0 = cube.minVect();
    const defaulttype::Vector3 & max_vect0 = cube.maxVect();
    const defaulttype::Vector3 & min_vect1 = other.cube.minVect();
    const defaulttype::Vector3 & max_vect1 = other.cube.maxVect();

    for(int i = 0 ; i < 3 ; ++i){
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect0[i] <= max_vect0[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        assert(min_vect1[i] <= max_vect1[i]);
        if(max_vect0[i] + alarmDist <= min_vect1[i] || max_vect1[i] + alarmDist <= min_vect0[i])
            return false;
    }

    return true;
}


using namespace sofa::defaulttype;
using namespace collision;

SOFA_DECL_CLASS(IncrSAP)

int IncrSAPClassSofaVector = core::RegisterObject("Collision detection using incremental sweep and prune")
        .addAlias( "IncrementalSAP" )
        .addAlias( "IncrementalSweepAndPrune" )
        .add< IncrSAP >( true )
        ;


} // namespace collision

} // namespace component

} // namespace sofa

