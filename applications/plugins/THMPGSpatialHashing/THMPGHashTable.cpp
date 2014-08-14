#include "THMPGHashTable.h"
#include <SofaBaseCollision/BaseIntTool.h>

using namespace sofa;
using namespace sofa::component::collision;

SReal THMPGHashTable::cell_size = (SReal)(0);
SReal THMPGHashTable::_alarmDist = (SReal)(0);
SReal THMPGHashTable::_alarmDistd2 = (SReal)(0);


void THMPGHashTable::init(int hashTableSize,core::CollisionModel *cm,SReal timeStamp){
    _cm = cm->getLast();
    resize(0);
    resize(hashTableSize);

    refersh(timeStamp);
}

void THMPGHashTable::refersh(SReal timeStamp){
    if(_timeStamp >= timeStamp)
        return;

    _timeStamp = timeStamp;

    sofa::component::collision::CubeModel* cube_model = dynamic_cast<sofa::component::collision::CubeModel*>(_cm->getPrevious());

    long int nb_added_elems = 0;
    int mincell[3];
    int maxcell[3];
    int movingcell[3];
    //SReal alarmDistd2 = intersectionMethod->getAlarmDistance()/((SReal)(2.0));

    //sofa::helper::AdvancedTimer::stepBegin("THMPGSpatialHashing : Hashing");

    Cube c(cube_model);

    for(;c.getIndex() < cube_model->getSize() ; ++c){
        ++nb_added_elems;
        const defaulttype::Vector3 & minVec = c.minVect();

        mincell[0] = std::floor((minVec[0] - _alarmDistd2)/cell_size);
        mincell[1] = std::floor((minVec[1] - _alarmDistd2)/cell_size);
        mincell[2] = std::floor((minVec[2] - _alarmDistd2)/cell_size);

        const defaulttype::Vector3 & maxVec = c.maxVect();
        maxcell[0] = std::floor((maxVec[0] + _alarmDistd2)/cell_size);
        maxcell[1] = std::floor((maxVec[1] + _alarmDistd2)/cell_size);
        maxcell[2] = std::floor((maxVec[2] + _alarmDistd2)/cell_size);

        for(movingcell[0] = mincell[0] ; movingcell[0] <= maxcell[0] ; ++movingcell[0]){
            for(movingcell[1] = mincell[1] ; movingcell[1] <= maxcell[1] ; ++movingcell[1]){
                for(movingcell[2] = mincell[2] ; movingcell[2] <= maxcell[2] ; ++movingcell[2]){
                    //sofa::helper::AdvancedTimer::stepBegin("THMPGSpatialHashing : addAndCollide");
                    (*this)(movingcell[0],movingcell[1],movingcell[2]).add(c/*.getExternalChildren().first*/,timeStamp);
                    //sofa::helper::AdvancedTimer::stepEnd("THMPGSpatialHashing : addAndCollide");
                }
            }
        }
    }
}

static bool checkIfCollisionIsDone(int i,int j,std::vector<int> * tab){
    for(unsigned int ii = 0 ; ii < tab[i].size() ; ++ii){
        if(tab[i][ii] == j)
            return true;
    }

    return false;
}

void THMPGHashTable::doCollision(THMPGHashTable & me,THMPGHashTable & other,sofa::core::collision::NarrowPhaseDetection * phase,SReal timeStamp,core::collision::ElementIntersector* ei,bool swap){
    sofa::core::CollisionModel* cm1,*cm2;
    cm1 = me.getCollisionModel();
    cm2 = other.getCollisionModel();

    assert(me._prime_size == other._prime_size);

    int size1,size2;
    if(swap){
        std::vector<int> * done_collisions = new std::vector<int>[cm2->getSize()];
        core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm2,cm1);
        ei->beginIntersect(cm2,cm1,output);

        for(int i = 0 ; i < me._prime_size ; ++i){
            if(me._table[i].updated(timeStamp) && other._table[i].updated(timeStamp)){
                std::vector<Cube> & vec_elems1 = me._table[i].getCollisionElems();
                std::vector<Cube> & vec_elems2 = other._table[i].getCollisionElems();

                size1 = vec_elems1.size();
                size2 = vec_elems2.size();

                for(int j = 0 ; j < size1 ; ++j){
                    for(int k = 0 ; k < size2 ; ++k){
                        if(!checkIfCollisionIsDone(vec_elems2[k].getIndex(),vec_elems1[j].getIndex(),done_collisions) && BaseIntTool::testIntersection(vec_elems2[k],vec_elems1[j],_alarmDist)){
                            ei->intersect(vec_elems2[k].getExternalChildren().first,vec_elems1[j].getExternalChildren().first,output);

                            done_collisions[vec_elems2[k].getIndex()].push_back(vec_elems1[j].getIndex());
                        }
                    }
                }
            }
        }

        delete[] done_collisions;
    }
    else{
        std::vector<int> * done_collisions = new std::vector<int>[cm1->getSize()];

        core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm1,cm2);
        ei->beginIntersect(cm1,cm2,output);

        for(int i = 0 ; i < me._prime_size ; ++i){
            if(me._table[i].updated(timeStamp) && other._table[i].updated(timeStamp)){
                std::vector<Cube> & vec_elems1 = me._table[i].getCollisionElems();
                std::vector<Cube> & vec_elems2 = other._table[i].getCollisionElems();

                size1 = vec_elems1.size();
                size2 = vec_elems2.size();

                for(int j = 0 ; j < size1 ; ++j){
                    for(int k = 0 ; k < size2 ; ++k){
                        if((!checkIfCollisionIsDone(vec_elems1[j].getIndex(),vec_elems2[k].getIndex(),done_collisions)) && BaseIntTool::testIntersection(vec_elems1[j],vec_elems2[k],_alarmDist)){
                            ei->intersect(vec_elems1[j].getExternalChildren().first,vec_elems2[k].getExternalChildren().first,output);

                            done_collisions[vec_elems1[j].getIndex()].push_back(vec_elems2[k].getIndex());
                        }
                    }
                }
            }
        }

        delete[] done_collisions;
    }
}


void THMPGHashTable::autoCollide(core::collision::NarrowPhaseDetection * phase,sofa::core::collision::Intersection * interMethod,SReal timeStamp){
    sofa::core::CollisionModel* cm = getCollisionModel();

    int size,sizem1;

    std::vector<int> * done_collisions = new std::vector<int>[cm->getSize()];
    core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm,cm);
    bool swap;
    sofa::core::collision::ElementIntersector * ei = interMethod->findIntersector(cm,cm,swap);
    ei->beginIntersect(cm,cm,output);

    for(int i = 0 ; i < _prime_size ; ++i){
        if(_table[i].needsCollision(timeStamp)){
            std::vector<Cube> & vec_elems = _table[i].getCollisionElems();

            size = vec_elems.size();
            sizem1 = size - 1;

            for(int j = 0 ; j < sizem1 ; ++j){
                for(int k = j + 1 ; k < size ; ++k){
                    if(!checkIfCollisionIsDone(vec_elems[j].getIndex(),vec_elems[k].getIndex(),done_collisions) && BaseIntTool::testIntersection(vec_elems[j],vec_elems[k],_alarmDist)){
                        ei->intersect(vec_elems[j].getExternalChildren().first,vec_elems[k].getExternalChildren().first,output);

                        done_collisions[vec_elems[j].getIndex()].push_back(vec_elems[k].getIndex());
                        //WARNING : we don't add the symetric done_collisions[vec_elems[k].getIndex()].push_back(vec_elems[j].getIndex()); because
                        //elements are added first in all cells they belong to, then next elements are added, so that if two elements share two same
                        //cells, one element will be first encountered in the both cells.
                    }
                }
            }
        }
    }

    delete[] done_collisions;
}


void THMPGHashTable::collide(THMPGHashTable & other,sofa::core::collision::NarrowPhaseDetection * phase,sofa::core::collision::Intersection * interMehtod,SReal timeStamp){
    sofa::core::CollisionModel* cm1,*cm2;
    cm1 = getCollisionModel();
    cm2 = other.getCollisionModel();

    if(!(cm1->canCollideWith(cm2)))
        return;

    THMPGHashTable * ptable1,*ptable2;

    if(cm1->getSize() <= cm2->getSize()){
        ptable1 = this;
        ptable2 = &other;
    }
    else{
        std::swap(cm1,cm2);
        ptable1 = &other;
        ptable2 = this;
    }

    bool swap;
    core::collision::ElementIntersector* ei = interMehtod->findIntersector(cm1,cm2,swap);

    if(ei == 0x0)
        return;

    doCollision(*ptable1,*ptable2,phase,timeStamp,ei,swap);
}



