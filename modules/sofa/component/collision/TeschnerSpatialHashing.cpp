#include <sofa/component/collision/TeschnerSpatialHashing.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/TetrahedronModel.h>
#include <sofa/component/collision/LineModel.h>

#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace collision
{

struct CannotInitializeCellSize : std::exception {
    virtual const char* what() const throw(){return "Cannot initialize cell size in TeschnerSpatialHashing because the scene doesn't contain any BaseMeshTopology";}
};


using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace collision;

SOFA_DECL_CLASS(TeschnerSpatialHashing)

int TeschnerSpatialHashingClass = core::RegisterObject("Collision detection using Teschner spatial hashing.")
        .add< TeschnerSpatialHashing >()
        ;

TeschnerSpatialHashing::TeschnerSpatialHashing(){
    _timeStamp = 0;
    _total_edges_length = 0.0;
    _nb_elems = 0;
    _nb_edges = 0x0;
    _params_initialized = false;
}

void TeschnerSpatialHashing::init(){
    reinit();
}


void TeschnerSpatialHashing::reinit(){
    _timeStamp = 0;
    _total_edges_length = 0.0;
    _nb_elems = 0;
    _nb_edges = 0x0;
    _params_initialized = false;
    _grid.clear();
}


template <class DataTypes>
void TeschnerSpatialHashing::sumEdgeLength_template(core::CollisionModel *cm){
    sofa::core::topology::BaseMeshTopology * bmt = cm->getContext()->get<sofa::core::topology::BaseMeshTopology>(sofa::core::objectmodel::BaseContext::Local);

    if(bmt == 0x0)
        return;

    sofa::core::behavior::MechanicalState<DataTypes> * mec = cm->getContext()->get<sofa::core::behavior::MechanicalState<DataTypes> >(sofa::core::objectmodel::BaseContext::Local);
    if(mec == 0)
        return;

    const typename MechanicalState<DataTypes>::VecCoord & coords = *(mec->getX());

    const sofa::core::topology::BaseMeshTopology::SeqEdges & seq_edges = bmt->getEdges();

    for(int i = 0 ; i < seq_edges.size() ; ++i)
        _total_edges_length += (DataTypes::getCPos(coords[seq_edges[i][0]]) - DataTypes::getCPos(coords[seq_edges[i][1]])).norm();

    _nb_edges += seq_edges.size();
}


void TeschnerSpatialHashing::sumEdgeLength(core::CollisionModel *cm){
    if(cm->getEnumType() == sofa::core::CollisionModel::TRIANGLE_TYPE)
        sumEdgeLength_template<sofa::component::collision::TriangleModel::DataTypes>(cm);
    else if(cm->getEnumType() == sofa::core::CollisionModel::TETRAHEDRON_TYPE)
        sumEdgeLength_template<sofa::component::collision::TetrahedronModel::DataTypes>(cm);
    else if(cm->getEnumType() == sofa::core::CollisionModel::LINE_TYPE)
        sumEdgeLength_template<sofa::component::collision::LineModel::DataTypes>(cm);
}


void TeschnerSpatialHashing::addCollisionModel(core::CollisionModel *cm){
    if(!_params_initialized){
        sumEdgeLength(cm->getLast());
        _nb_elems += cm->getLast()->getSize();
    }

    CubeModel * cube_model = dynamic_cast<CubeModel *>(cm->getLast()->getPrevious());

    cubeModels.push_back(cube_model);
}


void TeschnerSpatialHashing::endBroadPhase(){
    BroadPhaseDetection::endBroadPhase();

    if(!_params_initialized){
        if(_total_edges_length == 0){
            CannotInitializeCellSize c;
            throw c;
        }

        _cell_size = _total_edges_length/_nb_edges;
        _params_initialized = true;

        SReal* minBBox = new SReal[3];
        SReal* maxBBox = new SReal[3];
        sofa::simulation::getSimulation()->computeBBox(sofa::simulation::getSimulation()->GetRoot().get(),minBBox,maxBBox);

        std::cout<<"minBBox "<<minBBox[0]<<" "<<minBBox[1]<<" "<<minBBox[2]<<std::endl;
        std::cout<<"maxBBox "<<maxBBox[0]<<" "<<maxBBox[1]<<" "<<maxBBox[2]<<std::endl;
        std::cout <<"BIG SIZE "<<3 * ((maxBBox[0] - minBBox[0])/_cell_size) * ((maxBBox[1] - minBBox[1])/_cell_size) * ((maxBBox[2] - minBBox[2])/_cell_size)<<std::endl;
//        _grid.resize( 3 * ((maxBBox[0] - minBBox[0])/_cell_size) * ((maxBBox[1] - minBBox[1])/_cell_size) * ((maxBBox[2] - minBBox[2])/_cell_size));

        _grid.resize(10000);

        delete [] minBBox;
        delete [] maxBBox;
    }
}

void TeschnerSpatialHashing::beginNarrowPhase(){
    NarrowPhaseDetection::beginNarrowPhase();

    int mincell[3];
    int maxcell[3];
    int movingcell[3];

    sofa::helper::AdvancedTimer::stepBegin("TeschnerSpatialHashing : Hashing");
    for(int i = 0 ; i < cubeModels.size() ; ++i){
        //std::cout<<"Treating model "<<cubeModels[i]->getLast()->getName()<<std::endl;
        Cube c(cubeModels[i]);

        for(;c.getIndex() < cubeModels[i]->getSize() ; ++c){
            const Vector3 & minVec = c.minVect();

            mincell[0] = std::floor(minVec[0]/_cell_size);
            mincell[1] = std::floor(minVec[1]/_cell_size);
            mincell[2] = std::floor(minVec[2]/_cell_size);

            const Vector3 & maxVec = c.maxVect();
            maxcell[0] = std::floor(maxVec[0]/_cell_size);
            maxcell[1] = std::floor(maxVec[1]/_cell_size);
            maxcell[2] = std::floor(maxVec[2]/_cell_size);

            if(mincell[0] == maxcell[0] && mincell[1] == maxcell[1] && mincell[2] == maxcell[2]){
                sofa::helper::AdvancedTimer::stepBegin("TeschnerSpatialHashing : addAndCollide");
                _grid.addAndCollide(mincell[0],mincell[1],mincell[2],c.getExternalChildren().first,_timeStamp,this,intersectionMethod);
                sofa::helper::AdvancedTimer::stepEnd("TeschnerSpatialHashing : addAndCollide");
//                TeschnerCollisionSet & tset = _grid(mincell[0],mincell[1],mincell[2]);
//                tset.add(c.getExternalChildren().first,_timeStamp);
                //std::cout<<"\tadding elem at index "<<mincell[0]<<" "<<mincell[1]<<" "<<mincell[2]<<" and index "<<_grid.getIndex(mincell[0],mincell[1],mincell[2])<<std::endl;
                continue;
            }


            for(movingcell[0] = mincell[0] ; movingcell[0] <= maxcell[0] ; ++movingcell[0]){
                for(movingcell[1] = mincell[1] ; movingcell[1] <= maxcell[1] ; ++movingcell[1]){
                    for(movingcell[2] = mincell[2] ; movingcell[2] <= maxcell[2] ; ++movingcell[2]){
                        sofa::helper::AdvancedTimer::stepBegin("TeschnerSpatialHashing : addAndCollide");
                        _grid.addAndCollide(movingcell[0],movingcell[1],movingcell[2],c.getExternalChildren().first,_timeStamp,this,intersectionMethod);
                        sofa::helper::AdvancedTimer::stepEnd("TeschnerSpatialHashing : addAndCollide");
                        //std::cout<<"\tadding elem at index "<<movingcell[0]<<" "<<movingcell[1]<<" "<<movingcell[2]<<" and index "<<_grid.getIndex(movingcell[0],movingcell[1],movingcell[2])<<std::endl;
//                        TeschnerCollisionSet & tset = _grid(movingcell[0],movingcell[1],movingcell[2]);
//                        tset.add(c.getExternalChildren().first,_timeStamp);
                    }
                }
            }
        }
    }

    sofa::helper::AdvancedTimer::stepEnd("TeschnerSpatialHashing : Hashing");

//    sofa::helper::AdvancedTimer::stepBegin("TeschnerSpatialHashing : performCollision");
//    _grid.performCollision(this,this->intersectionMethod,_timeStamp);
//    sofa::helper::AdvancedTimer::stepEnd("TeschnerSpatialHashing : performCollision");
}

bool TeschnerSpatialHashing::keepCollisionBetween(core::CollisionModel *cm1, core::CollisionModel *cm2)
{
    if (!cm1->canCollideWith(cm2) || !cm2->canCollideWith(cm1))
    {
        return false;
    }

    return true;
}

}
}
}
