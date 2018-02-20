/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef COLLISIONPM_H
#define COLLISIONPM_H
#include "config.h"

#include <sofa/core/CollisionElement.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <unordered_map>
#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/hash.h>

namespace sofa
{

namespace component
{

namespace collision
{

    class CollidingPair{
    public:
        CollidingPair(){}

        /**
          *x and y value are used to identify this pair, the order is not important but the order of elem1_ and elem2_ is because
          *when using intersect funcion we run inter->intersect(elem1,elem2,output), so the intersect method must handle the order you give.
          */
        CollidingPair(const core::CollisionElementIterator & elem1_,const core::CollisionElementIterator & elem2_,core::collision::ElementIntersector * /*inter*/) :
            elem1(elem1_),elem2(elem2_)
        {
        }

        void init(const core::CollisionElementIterator & elem1_,const core::CollisionElementIterator & elem2_){
            assert(elem1_.getIndex() >= 0);
            assert(elem2_.getIndex() >= 0);
            elem1 = elem1_;
            elem2 = elem2_;
        }

        core::CollisionElementIterator elem1;
        core::CollisionElementIterator elem2;
    };


    class CollisionPairID{
    public:
        CollisionPairID(int x,int y){
            if(x < y){
                _id1 = x;
                _id2 = y;
            }
            else{
                _id1 = y;
                _id2 = x;
            }
        }

        bool operator<(const CollisionPairID & other)const{
            if(this->_id1 < other._id1)
                return true;
            if(this->_id1 > other._id1)
                return false;

            return this->_id2 < other._id2;
        }

    private:
        int _id1;
        int _id2;
    };

    class CollidingPM{
    public:
        struct CollModID{
            int enum_type;
            core::CollisionModel* sample;//just one collision model used to find the intersector further

            CollModID(){}
            CollModID(int id,core::CollisionModel* cm) : enum_type(id),sample(cm){}

            bool operator<(const CollModID & other)const{
                return this->enum_type < other.enum_type;
            }
        };


        typedef std::unordered_map<std::pair<int,int>,CollidingPair> umap_collision;


        CollidingPM(){
            for(int i = 0 ; i < sofa::core::CollisionModel::ENUM_TYPE_SIZE ; ++i){
                for(int j = 0 ; j < sofa::core::CollisionModel::ENUM_TYPE_SIZE ; ++j){
                    _order[i][j] = 0;
                    _intersectors[i][j] = 0x0;
                }
            }
        }

        void add(int a,int b,const core::CollisionElementIterator & elem1,const core::CollisionElementIterator & elem2){
            assert(elem1.getIndex() < elem1.getCollisionModel()->getSize());
            assert(elem2.getIndex() < elem2.getCollisionModel()->getSize());
            assert(elem1.getIndex() >= 0);
            assert(elem2.getIndex() >= 0);

            core::CollisionModel * cm1 = elem1.getCollisionModel();
            core::CollisionModel * cm2 = elem2.getCollisionModel();
            if(_order[cm1->getEnumType()][cm2->getEnumType()] == 2){//it means that cm1->getEnumType() == cm2->getEnumType()
                if(a < b)
                    _coll_pairs[cm1->getEnumType()][cm2->getEnumType()][std::make_pair(a,b)].init(elem1,elem2);
                else
                    _coll_pairs[cm2->getEnumType()][cm1->getEnumType()][std::make_pair(b,a)].init(elem2,elem1);
            }
            else if(_order[elem1.getCollisionModel()->getEnumType()][elem2.getCollisionModel()->getEnumType()] == 1){
                _coll_pairs[cm1->getEnumType()][cm2->getEnumType()][std::make_pair(a,b)].init(elem1,elem2);
            }
            else if(_order[elem1.getCollisionModel()->getEnumType()][elem2.getCollisionModel()->getEnumType()] == -1){
                _coll_pairs[cm2->getEnumType()][cm1->getEnumType()][std::make_pair(b,a)].init(elem2,elem1);
            }
        }

        void add(core::CollisionModel* cm,sofa::core::collision::Intersection * interMehtod){
            if((_addedCM.insert(CollModID(cm->getEnumType(),cm))).second){

                bool swap;
                core::collision::ElementIntersector * ei = interMehtod->findIntersector(cm,cm,swap);
                if(ei){
                    _order[cm->getEnumType()][cm->getEnumType()] = 2;
                    _intersectors[cm->getEnumType()][cm->getEnumType()] = ei;
                }

                for(std::set<CollModID>::iterator it = _addedCM.begin() ; it != _addedCM.end() ; ++it){
                    if(it->sample->getEnumType() == cm->getEnumType())
                        continue;

                    swap = false;                                        
                    ei = interMehtod->findIntersector(cm,it->sample,swap);

                    if(ei && swap){
                        _order[cm->getEnumType()][it->enum_type] = -1;
                        _order[it->enum_type][cm->getEnumType()] = 1;
                        _intersectors[it->enum_type][cm->getEnumType()] = _intersectors[cm->getEnumType()][it->enum_type] = ei;
                    }
                    else if(ei){
                        _order[cm->getEnumType()][it->enum_type] = 1;
                        _order[it->enum_type][cm->getEnumType()] = -1;
                        _intersectors[it->enum_type][cm->getEnumType()] = _intersectors[cm->getEnumType()][it->enum_type] = ei;
                    }
                }
            }
        }

        void remove(int a,int b,const core::CollisionElementIterator & elem1,const core::CollisionElementIterator & elem2){
            assert(elem1.getIndex() < elem1.getCollisionModel()->getSize());
            assert(elem2.getIndex() < elem2.getCollisionModel()->getSize());
            assert(elem1.getIndex() >= 0);
            assert(elem2.getIndex() >= 0);

            core::CollisionModel * cm1 = elem1.getCollisionModel();
            core::CollisionModel * cm2 = elem2.getCollisionModel();
            if(_order[cm1->getEnumType()][cm2->getEnumType()] == 0){
                return;
            }
            else if(_order[cm1->getEnumType()][cm2->getEnumType()] == 2){//it means that cm1->getEnumType() == cm2->getEnumType()
                if(a < b)
                    _coll_pairs[cm1->getEnumType()][cm2->getEnumType()].erase(std::make_pair(a,b));
                else
                    _coll_pairs[cm2->getEnumType()][cm1->getEnumType()].erase(std::make_pair(b,a));
            }
            else if(_order[elem1.getCollisionModel()->getEnumType()][elem2.getCollisionModel()->getEnumType()] == 1){
                _coll_pairs[cm1->getEnumType()][cm2->getEnumType()].erase(std::make_pair(a,b));
            }
            else if(_order[elem1.getCollisionModel()->getEnumType()][elem2.getCollisionModel()->getEnumType()] == -1){
                _coll_pairs[cm2->getEnumType()][cm1->getEnumType()].erase(std::make_pair(b,a));
            }
        }

        void clear(){
            _addedCM.clear();
            for(int i = 0 ; i < sofa::core::CollisionModel::ENUM_TYPE_SIZE ; ++i){
                for(int j = 0 ; j < sofa::core::CollisionModel::ENUM_TYPE_SIZE ; ++j){
                    _coll_pairs[i][j].clear();
//                    _order[i][j] = 0;
//                    _intersectors[i][j] = 0;
                }
            }
        }

        void intersect(core::collision::NarrowPhaseDetection * phase){            
            for(int i = 0 ; i < sofa::core::CollisionModel::ENUM_TYPE_SIZE ; ++i){
                for(int j = 0 ; j < sofa::core::CollisionModel::ENUM_TYPE_SIZE ; ++j){
                    if(_order[i][j] > 0){
                        core::collision::ElementIntersector * ei = _intersectors[i][j];
                        for(umap_collision::iterator it = _coll_pairs[i][j].begin() ; it != _coll_pairs[i][j].end() ; ++it){
                            core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(it->second.elem1.getCollisionModel(),it->second.elem2.getCollisionModel());
                            ei->beginIntersect(it->second.elem1.getCollisionModel(),it->second.elem2.getCollisionModel(),output);
                            ei->intersect(it->second.elem1,it->second.elem2,output);
                        }
                    }
                }
            }
        }            

    private:
        std::set<CollModID> _addedCM;//_addedCM are collision model TYPES
        int _order[sofa::core::CollisionModel::ENUM_TYPE_SIZE][sofa::core::CollisionModel::ENUM_TYPE_SIZE];
        umap_collision _coll_pairs[sofa::core::CollisionModel::ENUM_TYPE_SIZE][sofa::core::CollisionModel::ENUM_TYPE_SIZE];
        core::collision::ElementIntersector * _intersectors[sofa::core::CollisionModel::ENUM_TYPE_SIZE][sofa::core::CollisionModel::ENUM_TYPE_SIZE];
    };
}
}
}
#endif // COLLISIONPM_H
