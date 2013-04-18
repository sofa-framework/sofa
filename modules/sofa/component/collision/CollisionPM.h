#ifndef COLLISIONPM_H
#define COLLISIONPM_H
#include <sofa/core/CollisionElement.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <boost/unordered_map.hpp>

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
        CollidingPair(const core::CollisionElementIterator & elem1_,const core::CollisionElementIterator & elem2_,core::collision::ElementIntersector * inter) :
            elem1(elem1_),elem2(elem2_),intersector(inter)
        {
        }

        void init(const core::CollisionElementIterator & elem1_,const core::CollisionElementIterator & elem2_,core::collision::ElementIntersector * inter){
            assert(elem1_.getIndex() >= 0);
            assert(elem2_.getIndex() >= 0);
            elem1 = elem1_;
            elem2 = elem2_;
            assert(elem1.getIndex() >= 0);
            assert(elem2.getIndex() >= 0);
            intersector = inter;
        }

        int intersect(core::collision::NarrowPhaseDetection * phase){
            sofa::core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(elem1.getCollisionModel(),elem2.getCollisionModel());
            intersector->beginIntersect(elem1.getCollisionModel(),elem2.getCollisionModel(),output);
            return intersector->intersect(elem1,elem2,output);
        }

        core::CollisionElementIterator elem1;
        core::CollisionElementIterator elem2;
        core::collision::ElementIntersector * intersector;
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
        typedef boost::unordered::unordered_map<std::pair<int,int>,CollidingPair> umap_collision;

        void add(int a,int b,const core::CollisionElementIterator & elem1,const core::CollisionElementIterator & elem2,core::collision::ElementIntersector * inter){
            assert(elem1.getIndex() < elem1.getCollisionModel()->getSize());
            assert(elem2.getIndex() < elem2.getCollisionModel()->getSize());
            assert(elem1.getIndex() >= 0);
            assert(elem2.getIndex() >= 0);
            if(a < b)
                _coll_pairs[std::make_pair(a,b)].init(elem1,elem2,inter);
            else
                _coll_pairs[std::make_pair(b,a)].init(elem1,elem2,inter);
        }

        void remove(int a,int b){
            if(a < b)
                _coll_pairs.erase(std::make_pair(a,b));
            else
                _coll_pairs.erase(std::make_pair(b,a));
        }

        void clear(){
            _coll_pairs.clear();
        }

        void intersect(core::collision::NarrowPhaseDetection * phase){
            for(umap_collision::iterator it = _coll_pairs.begin() ; it != _coll_pairs.end() ; ++it){
                it->second.intersect(phase);
            }
        }

        int size(){
            return _coll_pairs.size();
        }

    private:
        umap_collision _coll_pairs;
    };
}
}
}
#endif // COLLISIONPM_H
