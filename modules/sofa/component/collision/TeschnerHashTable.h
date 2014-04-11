#include <vector>
#include <sofa/core/CollisionElement.h>
#include <boost/unordered/detail/util.hpp>
#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/NarrowPhaseDetection.h>
#include <sofa/helper/AdvancedTimer.h>
#include <boost/functional/hash.hpp>
#include <sofa/component/collision/CubeModel.h>


namespace sofa{

//class MirrorIntersector : public core::collision::ElementIntersector
//{
//public:
//    core::collision::ElementIntersector* intersector;

//    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
//    virtual bool canIntersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2)
//    {
//        return intersector->canIntersect(elem2, elem1);
//    }

//    /// Begin intersection tests between two collision models. Return the number of contacts written in the contacts vector.
//    /// If the given contacts vector is NULL, then this method should allocate it.
//    virtual int beginIntersect(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::DetectionOutputVector*& contacts)
//    {
//        return intersector->beginIntersect(model2, model1, contacts);
//    }

//    /// Compute the intersection between 2 elements. Return the number of contacts written in the contacts vector.
//    virtual int intersect(core::CollisionElementIterator elem1, core::CollisionElementIterator elem2, core::collision::DetectionOutputVector* contacts)
//    {
//        return intersector->intersect(elem2, elem1, contacts);
//    }

//    /// End intersection tests between two collision models. Return the number of contacts written in the contacts vector.
//    virtual int endIntersect(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::DetectionOutputVector* contacts)
//    {
//        return intersector->endIntersect(model2, model1, contacts);
//    }

//    virtual std::string name() const
//    {
//        return intersector->name() + std::string("<SWAP>");
//    }

//};


class TeschnerCollisionSet{
public:
    TeschnerCollisionSet() : _timeStamp(SReal(-1.0)){}

    void add(core::CollisionElementIterator elem,SReal timeStamp){
        if(_timeStamp < timeStamp){
            _timeStamp = timeStamp;
            _coll_elems.clear();
        }

        _coll_elems.push_back(elem);
    }

    void clearAndAdd(core::CollisionElementIterator elem,SReal timeStamp){
        if(_timeStamp != -1)
            _coll_elems.clear();

        _coll_elems.push_back(elem);
        _timeStamp = timeStamp;
    }

    bool needsCollision(SReal timestamp){
        if(_timeStamp < timestamp)
            return false;

        if(_coll_elems.size() < 2)
            return false;

        return true;
    }

    inline bool updated(SReal timeStamp)const{
        return _timeStamp >= timeStamp;
    }

    std::vector<core::CollisionElementIterator> & getCollisionElems(){
        return _coll_elems;
    }

    const std::vector<core::CollisionElementIterator> & getCollisionElems()const {
        return _coll_elems;
    }

    inline void clear(){
        _timeStamp = -1.0;
        _coll_elems.clear();
    }

private:
    SReal _timeStamp;
    std::vector<core::CollisionElementIterator> _coll_elems;
};

class TeschnerHashTable{
public:
    TeschnerHashTable(){
        _p1 = 73856093;
        _p2 = 19349663;
        _p3 = 83492791;
    }

    TeschnerHashTable(int size){
        _p1 = 73856093;
        _p2 = 19349663;
        _p3 = 83492791;

        resize(size);
    }

    void resize(int size){
        _size = size;
        _prime_size = boost::unordered::detail::next_prime(size);
        std::cout<<"PRIM SIZE "<<_prime_size<<std::endl;
        _table.resize(_prime_size);
    }

    void clear(){
        _size = 0;
        _table.clear();
    }

    inline long int getIndex(long int i,long int j,long int k)const{
        //int index = (i * _p1 ^ j * _p2 ^ k * _p3) % _prime_size;
        long int index = ((i * _p1) ^ (j * _p2) ^ (k * _p3)) % _prime_size;


        if(index < 0)
            index += _prime_size;

        return index;
    }

    inline TeschnerCollisionSet & operator()(long int i,long int j,long int k){
        //int index = (i * _p1 ^ j * _p2 ^ k * _p3) % _prime_size;
        long int index = ((i * _p1) ^ (j * _p2) ^ (k * _p3)) % _prime_size;
//        long int index = i*j*k % _prime_size;

        if(index < 0)
            index += _prime_size;

        return _table[index];
    }

    inline const TeschnerCollisionSet & operator()(long int i,long int j,long int k)const{
        //int index = (i * _p1 ^ j * _p2 ^ k * _p3) % _prime_size;
        long int index = ((i * _p1) ^ (j * _p2) ^ (k * _p3)) % _prime_size;
//        long int index = i*j*k % _prime_size;

        if(index < 0)
            index += _prime_size;

        return _table[index];
    }

    void addAndCollide(long int i,long int j,long int k,core::CollisionElementIterator elem,SReal timeStamp,core::collision::NarrowPhaseDetection * phase,sofa::core::collision::Intersection * interMehtod){
        TeschnerCollisionSet & tset = (*this)(i,j,k);
        std::vector<core::CollisionElementIterator> & vec_elems = tset.getCollisionElems();

        if(tset.updated(timeStamp)){
            int size = vec_elems.size();
            bool swap;
            core::collision::ElementIntersector* ei;
            sofa::core::CollisionModel* cm1,*cm2;

            cm1 = elem.getCollisionModel();

            for(int ii = 0 ; ii < size ; ++ii){
                cm2 = vec_elems[ii].getCollisionModel();

                if(!(cm1->canCollideWith(cm2)))
                    continue;

                sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable : find intersector and intersect");

//                sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable::performCollision : findIntersector");
                ei = interMehtod->findIntersector(cm1,cm2,swap);
//                sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable::performCollision : findIntersector");

                if(ei){
                    if(swap){
//                        sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable::performCollision : find output vector");
                        core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm2,cm1);
                        ei->beginIntersect(cm2,cm1,output);
//                        sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable::performCollision : find output vector");

                        ei->intersect(vec_elems[ii],elem,output);
                    }
                    else{
//                        sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable::performCollision : find output vector");
                        core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm1,cm2);
                        ei->beginIntersect(cm1,cm2,output);
//                        sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable::performCollision : find output vector");

                        ei->intersect(elem,vec_elems[ii],output);
                    }
                }

                sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable : find intersector and intersect");

            }

            vec_elems.push_back(elem);
            //std::cout<<"=====adding elem, final size "<<vec_elems.size()<<std::endl;
        }
        else{
            tset.clearAndAdd(elem,timeStamp);
        }

//        if(vec_elems.size() > 18){
//            std::cout<<"size of cell superior to 18========================================="<<std::endl;
//            for(int ii = 0 ; ii < vec_elems.size() ; ++ii){
//                std::cout<<"\tcollision model : "<<vec_elems[ii].getCollisionModel()<<" index : "<<vec_elems[ii].getIndex()<<std::endl;
////                sofa::component::collision::Cube c();
////                c.getVIterator()
//            }
//        }
    }

    void performCollision(core::collision::NarrowPhaseDetection * phase,sofa::core::collision::Intersection * interMehtod,SReal timeStamp){
        int sizem1;
        int size;
        bool swap;
        core::collision::ElementIntersector* ei;
        sofa::core::CollisionModel* cm1,*cm2;
        for(int i = 0 ; i < _prime_size ; ++i){
            if(_table[i].needsCollision(timeStamp)){
                std::vector<core::CollisionElementIterator> & vec_elems = _table[i].getCollisionElems();

                size = vec_elems.size();
                sizem1 = size - 1;

                for(int j = 0 ; j < sizem1 ; ++j){
                    cm1 = vec_elems[j].getCollisionModel();

                    for(int k = j + 1 ; k < size ; ++k){
                        cm2 = vec_elems[k].getCollisionModel();

                        if(!(cm1->canCollideWith(cm2)))
                            continue;

                        //sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable::performCollision : findIntersector");
                        ei = interMehtod->findIntersector(cm1,cm2,swap);
                        //sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable::performCollision : findIntersector");
                        if(ei){

                            if(swap){
                                //sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable::performCollision : find output vector");
                                core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm2,cm1);
                                ei->beginIntersect(cm2,cm1,output);
                                //sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable::performCollision : find output vector");

                                ei->intersect(vec_elems[k],vec_elems[j],output);
                            }
                            else{
                                //sofa::helper::AdvancedTimer::stepBegin("TeschnerHashTable::performCollision : find output vector");
                                core::collision::DetectionOutputVector*& output = phase->getDetectionOutputs(cm1,cm2);
                                ei->beginIntersect(cm1,cm2,output);
                                //sofa::helper::AdvancedTimer::stepEnd("TeschnerHashTable::performCollision : find output vector");

                                ei->intersect(vec_elems[j],vec_elems[k],output);
                            }
                        }
                    }
                }
            }
        }
    }



    virtual ~TeschnerHashTable(){
//        for(int i = 0 ; i < _intersector_garbage.size() ; ++i)
//            delete _intersector_garbage[i];
    }

    void showStats(SReal timeStamp)const{
        std::size_t nb_full_cell = 0;
        std::size_t nb_elems = 0;
        std::size_t max_elems_in_cell = 0;

        for(std::size_t i = 0 ; i < _table.size() ; ++i){
            if(_table[i].updated(timeStamp)){
                ++nb_full_cell;
                nb_elems += _table[i].getCollisionElems().size();
                if(_table[i].getCollisionElems().size() > max_elems_in_cell)
                    max_elems_in_cell = _table[i].getCollisionElems().size();
            }
        }

        SReal nb_elems_per_cell = (SReal)(nb_elems)/nb_full_cell;

        std::cout<<"TeschnerHashTableStats ============================="<<std::endl;
        std::cout<<"\tnb full cells "<<nb_full_cell<<std::endl;
        std::cout<<"\tnb elems per cell "<<nb_elems_per_cell<<std::endl;
        std::cout<<"\tmax exems found in a single cell "<<max_elems_in_cell<<std::endl;
        std::cout<<"\ttable size "<<_table.size()<<std::endl;
        std::cout<<"nb elems in hash table "<<nb_elems<<std::endl;
        std::cout<<"===================================================="<<std::endl;
    }

    SReal cell_size;
protected:
    boost::hash<std::pair<long int,long int> > _hash_func;
    long int _p1;
    long int _p2;
    long int _p3;
    long int _size;
    long int _prime_size;
    std::vector<TeschnerCollisionSet> _table;
    //core::collision::ElementIntersector* _intersectors[sofa::core::CollisionModel::ENUM_TYPE_SIZE][sofa::core::CollisionModel::ENUM_TYPE_SIZE];
    //std::vector<MirrorIntersector*> _intersector_garbage;
};

}
