#include "config.h"

#include <vector>
#include <sofa/core/CollisionElement.h>
#include <boost/version.hpp>
#if BOOST_VERSION < 106400
#include <boost/unordered/detail/util.hpp>
#else
#include <boost/unordered/detail/implementation.hpp>
#endif
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/helper/AdvancedTimer.h>
#include <boost/functional/hash.hpp>
#include <SofaBaseCollision/CubeModel.h>


#define CHECK_IF_ELLEMENT_EXISTS
namespace sofa{

class SOFA_THMPGSPATIALHASHING_API THMPGCollisionSet
{
public:
    THMPGCollisionSet() : _timeStamp(SReal(-1.0)){}

    inline void add(sofa::component::collision::Cube elem,SReal timeStamp){
        //sofa::helper::AdvancedTimer::stepBegin("THMPGCollisionSet : add");

        if(_timeStamp < timeStamp){
            _timeStamp = timeStamp;
            _coll_elems.clear();
        }
#ifndef CHECK_IF_ELLEMENT_EXISTS
        _coll_elems.push_back(elem);
#else
        unsigned int i;
        for(i = 0 ; i < _coll_elems.size() ; ++i){
            if(_coll_elems[i].getIndex() == elem.getIndex())
                break;
        }

        if(i == _coll_elems.size()){
            _coll_elems.push_back(elem);
        }
#endif
        //sofa::helper::AdvancedTimer::stepEnd("THMPGCollisionSet : add");
    }

    inline void clearAndAdd(sofa::component::collision::Cube elem,SReal timeStamp){
        if(_timeStamp != -1)
            _coll_elems.clear();

        _coll_elems.push_back(elem);
        _timeStamp = timeStamp;
    }

    inline bool needsCollision(SReal timestamp){
        if(_timeStamp < timestamp)
            return false;

        if(_coll_elems.size() < 2)
            return false;

        return true;
    }

    inline bool updated(SReal timeStamp)const{
        return _timeStamp >= timeStamp;
    }

    inline std::vector<sofa::component::collision::Cube> & getCollisionElems(){
        return _coll_elems;
    }

    inline const std::vector<sofa::component::collision::Cube> & getCollisionElems()const {
        return _coll_elems;
    }

    inline void clear(){
        _timeStamp = -1.0;
        _coll_elems.clear();
    }

private:
    SReal _timeStamp;
    std::vector<sofa::component::collision::Cube> _coll_elems;
};

#undef CHECK_IF_ELLEMENT_EXISTS

class SOFA_THMPGSPATIALHASHING_API THMPGHashTable{
protected:

//    struct CollisionElementPair{
//        CollisionElementPair(sofa::core::CollisionElementIterator & e1,sofa::core::CollisionElementIterator & e2){
//            if(e1.getCollisionModel() < e2.getCollisionModel()){
//                _e1 = e1;
//                _e2 = e2;
//            }
//            else if(e1.getCollisionModel() > e2.getCollisionModel()){
//                _e1 = e2;
//                _e2 = e1;
//            }
//            else if(e1.getIndex() < e2.getIndex()){
//                _e1 = e1;
//                _e2 = e2;
//            }
//            else{
//                _e1 = e2;
//                _e2 = e1;
//            }
//        }

//        bool operator<(const CollisionElementPair & other){
//            if(_e1.getCollisionModel() < other._e1.getCollisionModel()){
//                return true;
//            }
//            else if(_e1.getCollisionModel() > other._e1.getCollisionModel()){
//                return false;
//            }
//            else if(_e2.getCollisionModel() < other._e2.getCollisionModel()){
//                return true;
//            }
//            else if(_e2.getCollisionModel() > other._e2.getCollisionModel()){
//                return false;
//            }
//            else if(_e1.getIndex() < other._e1.getIndex()){
//                return true;
//            }
//            else if(_e1.getIndex() > other._e1.getIndex()){
//                return false;
//            }
//            else if(_e2.getIndex() < other._e2.getIndex()){
//                return true;
//            }
//            else{
//                return false;
//            }
//        }

//    private:
//        sofa::core::CollisionElementIterator _e1;
//        sofa::core::CollisionElementIterator _e2;
//    };

public:
    THMPGHashTable() : _cm(0x0),_timeStamp(-1.0){
        _p1 = 73856093;
        _p2 = 19349663;
        _p3 = 83492791;
    }

    THMPGHashTable(int hashTableSize,sofa::core::CollisionModel * cm,SReal timeStamp) : _cm(cm),_timeStamp(-1.0){
        _p1 = 73856093;
        _p2 = 19349663;
        _p3 = 83492791;

        init(hashTableSize,cm,timeStamp);
    }

    inline void resize(int size){
        _size = size;
        _prime_size = boost::unordered::detail::next_prime(size);
        _table.resize(_prime_size);
    }

    inline void clear(){
        _size = 0;
        _table.clear();
    }

    inline long int getIndex(long int i,long int j,long int k)const{
        long int index = ((i * _p1) ^ (j * _p2) ^ (k * _p3)) % _prime_size;

        if(index < 0)
            index += _prime_size;

        return index;
    }

    inline const THMPGCollisionSet & operator()(long int i,long int j,long int k)const{
        long int index = ((i * _p1) ^ (j * _p2) ^ (k * _p3)) % _prime_size;

        if(index < 0)
            index += _prime_size;

        return _table[index];
    }

    void autoCollide(core::collision::NarrowPhaseDetection * phase,sofa::core::collision::Intersection * interMethod,SReal timeStamp);

    void collide(THMPGHashTable & other,sofa::core::collision::NarrowPhaseDetection * phase,sofa::core::collision::Intersection * interMehtod,SReal timeStamp);

    virtual ~THMPGHashTable(){}

    void showStats(SReal timeStamp)const{
        int nb_full_cell = 0;
        int nb_elems = 0;
        unsigned int max_elems_in_cell = 0;

        for(unsigned int i = 0 ; i < _table.size() ; ++i){
            if(_table[i].updated(timeStamp)){
                ++nb_full_cell;
                nb_elems += _table[i].getCollisionElems().size();
                if(_table[i].getCollisionElems().size() > max_elems_in_cell)
                    max_elems_in_cell = _table[i].getCollisionElems().size();
            }
        }

        SReal nb_elems_per_cell = (SReal)(nb_elems)/nb_full_cell;

        std::cout<<"THMPGHashTableStats ============================="<<std::endl;
        std::cout<<"\tnb full cells "<<nb_full_cell<<std::endl;
        std::cout<<"\tnb elems per cell "<<nb_elems_per_cell<<std::endl;
        std::cout<<"\tmax exems found in a single cell "<<max_elems_in_cell<<std::endl;
        std::cout<<"\ttable size "<<_table.size()<<std::endl;
        std::cout<<"nb elems in hash table "<<nb_elems<<std::endl;
        std::cout<<"===================================================="<<std::endl;
    }

    void init(int hashTableSize, core::CollisionModel *cm, SReal timeStamp);

    void refersh(SReal timeStamp);

    inline bool initialized()const{
        return _cm != 0x0;
    }

    inline core::CollisionModel* getCollisionModel()const{
        return _cm;
    }

    static SReal cell_size;

    inline THMPGCollisionSet & operator()(long int i,long int j,long int k){
        long int index = ((i * _p1) ^ (j * _p2) ^ (k * _p3)) % _prime_size;

        if(index < 0)
            index += _prime_size;

        return _table[index];
    }

    inline static void setAlarmDistance(SReal alarmDist){
        _alarmDist = alarmDist;
        _alarmDistd2 = alarmDist/2.0;
    }

protected:

    static void doCollision(THMPGHashTable &me, THMPGHashTable &other, core::collision::NarrowPhaseDetection *phase, SReal timeStamp, core::collision::ElementIntersector *ei, bool swap);


    sofa::core::CollisionModel * _cm;
    boost::hash<std::pair<long int,long int> > _hash_func;
    long int _p1;
    long int _p2;
    long int _p3;
    long int _size;
    long int _prime_size;
    std::vector<THMPGCollisionSet> _table;
    static SReal _alarmDist;
    static SReal _alarmDistd2;
    SReal _timeStamp;
};

}
