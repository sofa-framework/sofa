#include "THMPGSpatialHashing.h"
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaBaseCollision/OBBModel.h>

#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

struct CannotInitializeCellSize : std::exception {
    virtual const char* what() const throw(){return "Cannot initialize cell size in THMPGSpatialHashing";}
};


using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace collision;

SOFA_DECL_CLASS(THMPGSpatialHashing)

int THMPGSpatialHashingClass = core::RegisterObject("Collision detection using THMPG spatial hashing.")
        .add< THMPGSpatialHashing >()
        ;

THMPGSpatialHashing::THMPGSpatialHashing(){
    _max_cm_size = 0;
    _timeStamp = 0;
    _total_edges_length = 0.0;
    _nb_elems = 0;
    _nb_edges = 0x0;
    _params_initialized = false;
}

void THMPGSpatialHashing::init(){
    reinit();
}


void THMPGSpatialHashing::reinit(){
    _timeStamp = 0;
    _total_edges_length = 0.0;
    _nb_elems = 0;
    _nb_edges = 0x0;
    _params_initialized = false;
    _grid.clear();
}


template <class DataTypes>
void THMPGSpatialHashing::sumEdgeLength_template(core::CollisionModel *cm){
    sofa::core::topology::BaseMeshTopology * bmt = cm->getContext()->get<sofa::core::topology::BaseMeshTopology>(sofa::core::objectmodel::BaseContext::Local);

    if(bmt == 0x0)
        return;

    sofa::core::behavior::MechanicalState<DataTypes> * mec = cm->getContext()->get<sofa::core::behavior::MechanicalState<DataTypes> >(sofa::core::objectmodel::BaseContext::Local);
    if(mec == 0)
        return;

    const typename sofa::core::behavior::MechanicalState<DataTypes>::VecCoord & coords = mec->read(core::ConstVecCoordId::position())->getValue();

    const sofa::core::topology::BaseMeshTopology::SeqEdges & seq_edges = bmt->getEdges();

    for(unsigned int i = 0 ; i < seq_edges.size() ; ++i){
        //std::cout<<"one edge length "<< (DataTypes::getCPos(coords[seq_edges[i][0]]) - DataTypes::getCPos(coords[seq_edges[i][1]])).norm()<<std::endl;
        _total_edges_length += (DataTypes::getCPos(coords[seq_edges[i][0]]) - DataTypes::getCPos(coords[seq_edges[i][1]])).norm();
    }

    _nb_edges += seq_edges.size();
}


void THMPGSpatialHashing::sumEdgeLength(core::CollisionModel *cm){
    if(cm->getEnumType() == sofa::core::CollisionModel::TRIANGLE_TYPE)
        sumEdgeLength_template<sofa::component::collision::TriangleModel::DataTypes>(cm);
//    else if(cm->getEnumType() == sofa::core::CollisionModel::TETRAHEDRON_TYPE)
//        sumEdgeLength_template<sofa::component::collision::TetrahedronModel::DataTypes>(cm);
    else if(cm->getEnumType() == sofa::core::CollisionModel::LINE_TYPE)
        sumEdgeLength_template<sofa::component::collision::LineModel::DataTypes>(cm);
    else if(cm->getEnumType() == sofa::core::CollisionModel::SPHERE_TYPE){
        const sofa::component::collision::SphereModel * sphm = static_cast<SphereModel *>(cm);
        for(int i = 0 ; i < sphm->getSize() ; ++i){
            _total_edges_length += (SReal)(2) * sphm->getRadius(i);
        }
        _nb_edges += sphm->getSize();
    }
    else if(cm->getEnumType() == sofa::core::CollisionModel::OBB_TYPE){
        const OBBModel * obbm = static_cast<OBBModel *>(cm);
        for(int i = 0 ; i < obbm->getSize() ; ++i){
            const OBBModel::Coord & extents = obbm->extents(i);
            for(int j = 0 ; j < 3 ; ++j){
                _total_edges_length += (SReal)(2)*extents[j];
            }
        }
        _nb_edges += (SReal)(3) * obbm->getSize();
    }
}


void THMPGSpatialHashing::endBroadPhase(){
    BroadPhaseDetection::endBroadPhase();

    if(!_params_initialized){
        if(_total_edges_length == 0){
            CannotInitializeCellSize c;
            throw c;
        }

        _cell_size = /*0.2 **/ _total_edges_length/_nb_edges;
        THMPGHashTable::cell_size = _cell_size;
        THMPGHashTable::setAlarmDistance(intersectionMethod->getAlarmDistance());
        _params_initialized = true;

//        std::cout<<"cell size "<<_cell_size<<std::endl;
//        std::cout<<"nb elems "<<_nb_elems<<std::endl;
    }
}

void THMPGSpatialHashing::beginNarrowPhase(){
    NarrowPhaseDetection::beginNarrowPhase();
}

void THMPGSpatialHashing::addCollisionModel(core::CollisionModel *cm)
{
   //sofa::helper::AdvancedTimer::stepBegin("THMPGSpatialHashing::addCollisionModel");

    if(!_params_initialized){
        sumEdgeLength(cm->getLast());
        _nb_elems += cm->getLast()->getSize();

        if(_max_cm_size < cm->getLast()->getSize())
            _max_cm_size = cm->getLast()->getSize();
    }

    if (cm->isSimulated() && cm->getLast()->canCollideWith(cm->getLast()))
    {
        // self collision
        //sout << "Test broad phase Self "<<cm->getLast()->getName()<<sendl;
        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm->getLast(), cm->getLast(), swapModels);
        if (intersector != NULL)
                cmPairs.push_back(std::make_pair(cm, cm));

    }
    for (sofa::helper::vector<core::CollisionModel*>::iterator it = _collisionModels.begin(); it != _collisionModels.end(); ++it)
    {
        core::CollisionModel* cm2 = *it;

        if (!cm->isSimulated() && !cm2->isSimulated())
        {
            continue;
        }

        if (!keepCollisionBetween(cm->getLast(), cm2->getLast()))
            continue;

        bool swapModels = false;
        core::collision::ElementIntersector* intersector = intersectionMethod->findIntersector(cm->getLast(), cm2->getLast(), swapModels);
        if (intersector == NULL)
            continue;

        core::CollisionModel* cm1 = (swapModels?cm2:cm);
        cm2 = (swapModels?cm:cm2);

        // // Here we assume multiple root elements are present in both models
        // bool collisionDetected = false;
        // core::CollisionElementIterator begin1 = cm->begin();
        // core::CollisionElementIterator end1 = cm->end();
        // core::CollisionElementIterator begin2 = cm2->begin();
        // core::CollisionElementIterator end2 = cm2->end();
        // for (core::CollisionElementIterator it1 = begin1; it1 != end1; ++it1)
        // {
        //     for (core::CollisionElementIterator it2 = begin2; it2 != end2; ++it2)
        //     {
        //         //if (!it1->canCollideWith(it2)) continue;
        //         if (intersector->canIntersect(it1, it2))
        //         {
        //             collisionDetected = true;
        //             break;
        //         }
        //     }
        //     if (collisionDetected) break;
        // }
        // if (collisionDetected)

        // Here we assume a single root element is present in both models
        core::collision::ElementIntersector* AABBintersector = intersectionMethod->findIntersector(cm->getFirst(), cm2->getFirst(), swapModels);

        if (AABBintersector->canIntersect(cm1->getFirst()->begin(), cm2->getFirst()->begin()))
        {
            //sout << "Broad phase "<<cm1->getLast()->getName()<<" - "<<cm2->getLast()->getName()<<sendl;
            cmPairs.push_back(std::make_pair(cm1, cm2));
        }
    }
    _collisionModels.push_back(cm);

    //sofa::helper::AdvancedTimer::stepEnd("THMPGSpatialHashing::addCollisionModel");
}

bool THMPGSpatialHashing::keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2)
{
    if (!cm1->canCollideWith(cm2) || !cm2->canCollideWith(cm1))
    {
        return false;
    }

    return true;
}

void THMPGSpatialHashing::addCollisionPair (const std::pair<core::CollisionModel*, core::CollisionModel*>& coll_pair){
    //sofa::helper::AdvancedTimer::stepBegin("THMPGSpatialHashing::addCollisionPair");

    core::CollisionModel* cm1 = coll_pair.first->getLast();
    core::CollisionModel* cm2 = coll_pair.second->getLast();

    THMPGHashTable & t1 = _hash_tables[cm1];
    if(!t1.initialized())
        t1.init(_max_cm_size * 3 ,cm1,_timeStamp);
    else
        t1.refersh(_timeStamp);

    //t1.showStats(_timeStamp);
    if(cm1 == cm2){
        t1.autoCollide(this,intersectionMethod,_timeStamp);
        //sofa::helper::AdvancedTimer::stepEnd("THMPGSpatialHashing::addCollisionPair");

        return;
    }

    THMPGHashTable & t2 = _hash_tables[cm2];
    if(!t2.initialized())
        t2.init(_max_cm_size * 3,cm2,_timeStamp);
    else
        t2.refersh(_timeStamp);

    t1.collide(t2,this,intersectionMethod,_timeStamp);
    //t2.showStats(_timeStamp);
    //sofa::helper::AdvancedTimer::stepEnd("THMPGSpatialHashing::addCollisionPair");
}

}
}
}
