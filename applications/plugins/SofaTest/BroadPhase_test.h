#ifndef SOFA_STANDARDTEST_BroadPhase_test_H
#define SOFA_STANDARDTEST_BroadPhase_test_H

#include <SofaMeshCollision/DirectSAP.h>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/RandomGenerator.h>

#include <SofaComponentMain/init.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseCollision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include <plugins/SceneCreator/SceneCreator.h>

//#include <plugins/Flexible/deformationMapping/DistanceMapping.h>
//#include <plugins/Flexible/deformationMapping/DistanceFromTargetMapping.h>

#include <sofa/simulation/common/Simulation.h>
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaMeshCollision/MeshIntTool.h>
#include <sofa/helper/vector.h>
#include <SofaMeshCollision/IncrSAP.h>
#include <sofa/core/CollisionModel.h>
#include <stdlib.h>
//#include <sofa/component/collision/TeschnerSpatialHashing.h>


struct MyBox{

    MyBox(){}

    MyBox(sofa::component::collision::Cube cube_) : cube(cube_){}

    SReal squaredDistance(const MyBox & other)const{
        const sofa::defaulttype::Vector3 & min_vect0 = cube.minVect();
        const sofa::defaulttype::Vector3 & max_vect0 = cube.maxVect();
        const sofa::defaulttype::Vector3 & min_vect1 = other.cube.minVect();
        const sofa::defaulttype::Vector3 & max_vect1 = other.cube.maxVect();

        SReal temp;
        SReal dist2 = 0;

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

    void show()const{
        std::cout<<"collision model "<<cube.getCollisionModel()->getLast()<<" index "<<cube.getExternalChildren().first.getIndex()<<std::endl;
        std::cout<<"\tminBBox "<<cube.minVect()<<std::endl;
        std::cout<<"\tmaxBBox "<<cube.maxVect()<<std::endl;
    }

    sofa::component::collision::Cube cube;
};

template <class BroadPhase>
struct BroadPhaseTest: public ::testing::Test{
    static double getExtent(){return 1.2;}

    static bool randSparse();
    static bool randDense();
    static bool randTest3();

    static bool randTest(int seed,int nb1,int nb2,const sofa::defaulttype::Vector3 & min,const sofa::defaulttype::Vector3 & max);
};

struct InitIntersection{
    InitIntersection(sofa::component::collision::NewProximityIntersection::SPtr & prox,SReal alarmDist_){
        prox->setAlarmDistance(alarmDist_);
    }
};

//static bool goodBoundingTree(sofa::core::CollisionModel * cm){
//    sofa::component::collision::CubeModel * cbm = dynamic_cast<sofa::component::collision::CubeModel *>(cm->getFirst());
//    sofa::component::collision::Cube c(cbm);
//    const Vector3 & min = c.minVect();
//    const Vector3 & max = c.maxVect();

//    cbm = dynamic_cast<sofa::component::collision::CubeModel* >(cm->getFirst()->getNext());
//    sofa::component::collision::Cube c2(cbm);
//    while(c2.getIndex() < cbm->getSize()){
//        const Vector3 & min2 = c2.minVect();
//        const Vector3 & max2 = c2.maxVect();

//        for(int i = 0 ; i < 3 ; ++i){
//            if(min2[i] < min[i] || max2[i] > max[i])
//                return false;
//        }

//        ++c2;
//    }

//    return true;
//}

//intersection method used for the narrow phase
sofa::component::collision::NewProximityIntersection::SPtr proxIntersection = sofa::core::objectmodel::New<sofa::component::collision::NewProximityIntersection>();
InitIntersection initIntersection(proxIntersection,0);
double alarmDist = proxIntersection->getAlarmDistance();


//GENERAL FUNCTIONS
template<class Detection>
bool genTest(sofa::core::CollisionModel * cm1,sofa::core::CollisionModel * cm2,Detection & col_detection);

static sofa::defaulttype::Vector3 randVect(const sofa::defaulttype::Vector3 & min,const sofa::defaulttype::Vector3 & max, int seed=0);

void getMyBoxes(sofa::core::CollisionModel * cm,std::vector<MyBox> & my_boxes){
    sofa::component::collision::CubeModel * cbm = dynamic_cast<sofa::component::collision::CubeModel*>(cm->getLast()->getPrevious());
    assert(cbm != 0x0);

    for(int i = 0 ; i < cbm->getSize() ; ++i)
        my_boxes.push_back(MyBox(sofa::component::collision::Cube(cbm,i)));
}

sofa::component::collision::OBBModel::SPtr makeOBBModel(const std::vector<sofa::defaulttype::Vector3> & p,sofa::simulation::Node::SPtr &father,double default_extent);

void randMoving(sofa::core::CollisionModel* cm,const sofa::defaulttype::Vector3 & min_vect,const sofa::defaulttype::Vector3 & max_vect, int seed){
    sofa::component::collision::OBBModel * obbm = dynamic_cast<sofa::component::collision::OBBModel*>(cm->getLast());
    MechanicalObjectRigid3* dof = dynamic_cast<MechanicalObjectRigid3*>(obbm->getMechanicalState());

    sofa::core::objectmodel::Data<MechanicalObjectRigid3::VecCoord> & dpositions = *dof->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //Editting the velocity of the OBB
    sofa::core::objectmodel::Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *dof->write( sofa::core::VecId::velocity() );
    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    sofa::helper::RandomGenerator randomGenerator;
    randomGenerator.initSeed(seed);

    for(int i = 0 ; i < dof->getSize() ; ++i){
        if(randomGenerator.random<double>() < RAND_MAX/2.0){//make it move !
            velocities[i] = sofa::defaulttype::Vector3(1,1,1);//velocity is used only to know if a primitive moves, its direction is not important
            positions[i] = sofa::defaulttype::Rigid3Types::Coord(randVect(min_vect,max_vect, seed),sofa::defaulttype::Quaternion(0,0,0,1));
        }
    }

    dvelocities.endEdit();
    dpositions.endEdit();

    cm->computeBoundingTree(0);
}

//CLASS FUNCTIONS

struct CItCompare{
    void rearrenge(const std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> & p1,const std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> & p2,sofa::core::CollisionElementIterator & e11,
                    sofa::core::CollisionElementIterator & e12,sofa::core::CollisionElementIterator & e21,sofa::core::CollisionElementIterator & e22)const{
        if(p1.first.getCollisionModel()->getLast() == p1.second.getCollisionModel()->getLast()){
            if(p1.first.getIndex() < p1.second.getIndex()){
                e11 = p1.first;
                e12 = p1.second;
            }
            else{
                e12 = p1.first;
                e11 = p1.second;
            }
        }
        else if(p1.first.getCollisionModel()->getLast() < p1.second.getCollisionModel()->getLast()){
            e11 = p1.first;
            e12 = p1.second;
        }
        else{
            e12 = p1.first;
            e11 = p1.second;
        }

        if(p2.first.getCollisionModel()->getLast() == p2.second.getCollisionModel()->getLast()){
            if(p2.first.getIndex() < p2.second.getIndex()){
                e21 = p2.first;
                e22 = p2.second;
            }
            else{
                e22 = p2.first;
                e21 = p2.second;
            }
        }
        else if(p2.first.getCollisionModel()->getLast() < p2.second.getCollisionModel()->getLast()){
            e21 = p2.first;
            e22 = p2.second;
        }
        else{
            e22 = p2.first;
            e21 = p2.second;
        }
    }

    bool operator()(const std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> & p1,const std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> & p2)const{
        sofa::core::CollisionElementIterator e11,e12,e21,e22;
        rearrenge(p1,p2,e11,e12,e21,e22);

        if(e11.getCollisionModel()->getLast() != e21.getCollisionModel()->getLast())
            return e11.getCollisionModel()->getLast() < e21.getCollisionModel()->getLast();

        if(e12.getCollisionModel()->getLast() != e22.getCollisionModel()->getLast())
            return e12.getCollisionModel()->getLast() < e22.getCollisionModel()->getLast();

        if(e11.getIndex() != e21.getIndex())
            return e11.getIndex() < e21.getIndex();

        return e12.getIndex() < e22.getIndex();
    }

    bool same(const std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> & p1,const std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> & p2)const{
        sofa::core::CollisionElementIterator e11,e12,e21,e22;
        rearrenge(p1,p2,e11,e12,e21,e22);

        return e11.getCollisionModel()->getLast() == e21.getCollisionModel()->getLast() && e12.getCollisionModel()->getLast() == e22.getCollisionModel()->getLast() &&
                e11.getIndex() == e21.getIndex() && e12.getIndex() == e22.getIndex();
    }
};

template<class Detection>
bool GENTest(sofa::core::CollisionModel * cm1,sofa::core::CollisionModel * cm2,Detection & col_detection){
//    assert(goodBoundingTree((cm1)));
//    assert(goodBoundingTree((cm2)));
    cm1->setSelfCollision(true);
    cm2->setSelfCollision(true);

    col_detection.setIntersectionMethod(proxIntersection.get());

//    col_detection.addCollisionModel(cm1);
//    if(cm2 != 0x0)
//        col_detection.addCollisionModel(cm2);

    std::vector<MyBox> boxes;
    std::vector<std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> > brutInter;

    getMyBoxes(cm1,boxes);
    if(cm2 != 0x0)
        getMyBoxes(cm2,boxes);

    //cm1 self intersections
    for(unsigned int i = 0 ; i < boxes.size() ; ++i){
        for(unsigned int j = i + 1 ; j < boxes.size() ; ++j){
//            std::cout<<"colliding models "<<boxes[i].cube.getCollisionModel()->getLast()<<" "<<boxes[j].cube.getCollisionModel()->getLast()<<std::endl;
//            std::cout<<"colliding indices "<<boxes[i].cube.getIndex()<<" "<<boxes[j].cube.getIndex()<<std::endl;
//            std::cout<<"min/max vect"<<std::endl;
//            boxes[i].show();
//            boxes[j].show();
            if(boxes[i].squaredDistance(boxes[j]) <= alarmDist * alarmDist){
                brutInter.push_back(std::make_pair((sofa::core::CollisionElementIterator)(boxes[i].cube),(sofa::core::CollisionElementIterator)(boxes[j].cube)));
//                std::cout<<"\tCOLLIDING"<<std::endl;
//                std::cout<<"boxi"<<std::endl;
//                boxes[i].show();
//                std::cout<<"boxj"<<std::endl;
//                boxes[j].show();
            }
            else{
                //std::cout<<"\tNOT"<<std::endl;
            }
        }
        //std::cout<<"=========>>"<<std::endl;
    }

//    std::cout<<"SORTED BRUTE"<<std::endl;
    CItCompare c;
    std::sort(brutInter.begin(),brutInter.end(),c);
//    for(unsigned int i = 0 ; i < brutInter.size() ; ++i){
//        std::cout<<"colliding models "<<brutInter[i].first.getCollisionModel()->getLast()<<" "<<brutInter[i].second.getCollisionModel()->getLast()<<std::endl;
//        std::cout<<"colliding indices "<<brutInter[i].first.getIndex()<<" "<<brutInter[i].second.getIndex()<<std::endl;
//    }
//    std::cout<<"========SORTED BRUTE"<<std::endl;

    col_detection.beginBroadPhase();
    col_detection.addCollisionModel(cm1->getFirst());
    if(cm2)
        col_detection.addCollisionModel(cm2->getFirst());

    col_detection.endBroadPhase();
    col_detection.beginNarrowPhase();
    col_detection.addCollisionPairs(col_detection.getCollisionModelPairs());
    col_detection.endNarrowPhase();

    std::vector<std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> > broadPhaseInter;

    sofa::helper::vector<sofa::core::collision::DetectionOutput> * res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm1,cm1));
    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            broadPhaseInter.push_back(((*res)[i]).elem);


    res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm1,cm2));

    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            broadPhaseInter.push_back(((*res)[i]).elem);

    res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm2,cm1));

    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            broadPhaseInter.push_back(((*res)[i]).elem);

    res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm2,cm2));
    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            broadPhaseInter.push_back(((*res)[i]).elem);

    std::sort(broadPhaseInter.begin(),broadPhaseInter.end(),c);

    col_detection.endNarrowPhase();

    if(brutInter.size() != broadPhaseInter.size()){
        std::cout<<"BRUT FORCE PAIRS"<<std::endl;
        for(unsigned int j = 0 ; j < brutInter.size() ; ++j){
            std::cout<<brutInter[j].first.getCollisionModel()->getLast()<<" "<<brutInter[j].second.getCollisionModel()->getLast()<<std::endl;
            std::cout<<brutInter[j].first.getIndex()<<" "<<brutInter[j].second.getIndex()<<std::endl;
            std::cout<<"=="<<std::endl;
        }

        std::cout<<"=========BROAD PHASE PAIRS"<<std::endl;
        for(unsigned int j = 0 ; j < broadPhaseInter.size() ; ++j){
            std::cout<<"alarmDist "<<alarmDist<<std::endl;
            std::cout<<broadPhaseInter[j].first.getCollisionModel()->getLast()<<" "<<broadPhaseInter[j].second.getCollisionModel()->getLast()<<std::endl;
            std::cout<<broadPhaseInter[j].first.getIndex()<<" "<<broadPhaseInter[j].second.getIndex()<<std::endl;
        }

        std::cout<<"want to show::::::::::"<<std::endl;
        for(size_t i = 0 ; i < boxes.size() ; ++i){
            boxes[i].show();
        }
        std::cout<<"=="<<std::endl;

        return false;
    }

    unsigned int i;
    for(i = 0 ; i < brutInter.size() ; ++i)
        if(!c.same(brutInter[i],broadPhaseInter[i]))
            break;

    if(i < brutInter.size()){
        std::cout<<"BRUT FORCE PAIRS"<<std::endl;
        for(unsigned int j = 0 ; j < brutInter.size() ; ++j){
            std::cout<<brutInter[j].first.getCollisionModel()->getLast()<<" "<<brutInter[j].second.getCollisionModel()->getLast()<<std::endl;
            std::cout<<brutInter[j].first.getIndex()<<" "<<brutInter[j].second.getIndex()<<std::endl;
            std::cout<<"=="<<std::endl;
        }

        std::cout<<"=========BROAD PHASE PAIRS"<<std::endl;
        for(unsigned int j = 0 ; j < broadPhaseInter.size() ; ++j){
            std::cout<<broadPhaseInter[j].first.getCollisionModel()->getLast()<<" "<<broadPhaseInter[j].second.getCollisionModel()->getLast()<<std::endl;
            std::cout<<broadPhaseInter[j].first.getIndex()<<" "<<broadPhaseInter[j].second.getIndex()<<std::endl;
            std::cout<<"=="<<std::endl;
        }

        std::cout<<"want to show::::::::::"<<std::endl;
        for(size_t i = 0 ; i < boxes.size() ; ++i){
            boxes[i].show();
        }
        std::cout<<"=="<<std::endl;

        return false;
    }

    return true;
}


sofa::component::collision::OBBModel::SPtr makeOBBModel(const std::vector<sofa::defaulttype::Vector3> & p,sofa::simulation::Node::SPtr &father,double default_extent){
    int n = p.size();
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr obb = father->createChild("obb");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr obbDOF = sofa::core::objectmodel::New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    obbDOF->resize(n);
    sofa::core::objectmodel::Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    for(int i = 0 ; i < n ; ++i)
        positions[i] = sofa::defaulttype::Rigid3Types::Coord(p[i],sofa::defaulttype::Quaternion(0,0,0,1));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    sofa::core::objectmodel::Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *obbDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    for(int i = 0 ; i < n ; ++i)
        velocities[i] = sofa::defaulttype::Vector3(0,0,0);
    dvelocities.endEdit();


    obb->addObject(obbDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::OBBModel::SPtr obbCollisionModel = sofa::core::objectmodel::New<sofa::component::collision::OBBModel >();
    obb->addObject(obbCollisionModel);

    //editting the OBBModel
    sofa::component::collision::OBBModel::Real & def_ext = *(obbCollisionModel->default_ext.beginEdit());
    def_ext = default_extent;

    obbCollisionModel->default_ext.endEdit();

    obbCollisionModel->init();
//    Data<sofa::component::collision::OBBModel::VecCoord> & dVecCoord = obbCollisionModel->writeExtents();
//    sofa::component::collision::OBBModel::VecCoord & vecCoord = *(dVecCoord.beginEdit());
//dVecCoord.endEdit();
    obbCollisionModel->computeBoundingTree(0);

    //std::cout<<"the proximity "<<obbCollisionModel->getProximity()<<std::endl;
    return obbCollisionModel;
}

sofa::defaulttype::Vector3 randVect(const sofa::defaulttype::Vector3 & min,const sofa::defaulttype::Vector3 & max, int seed){
    sofa::defaulttype::Vector3 ret;
    sofa::defaulttype::Vector3 extents = max - min;

    sofa::helper::RandomGenerator randomGenerator;
    randomGenerator.initSeed((unsigned int)time(NULL));

    for(int i = 0 ; i < 3 ; ++i){
        ret[i] = ((randomGenerator.random<double>())/RAND_MAX) * extents[i] + min[i];
    }

    return ret;
}


template <class BroadPhase>
bool BroadPhaseTest<BroadPhase>::randTest(int seed, int nb1, int nb2, const sofa::defaulttype::Vector3& min, const sofa::defaulttype::Vector3& max){

    std::vector<sofa::defaulttype::Vector3> firstCollision;
    std::vector<sofa::defaulttype::Vector3> secondCollision;

    for(int i = 0 ; i < nb1 ; ++i)
        firstCollision.push_back(randVect(min,max,seed));

    for(int i = 0 ; i < nb2 ; ++i)
        secondCollision.push_back(randVect(min,max,seed));

    sofa::simulation::Node::SPtr scn = sofa::core::objectmodel::New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbm1,obbm2;
    obbm1 = makeOBBModel(firstCollision,scn,getExtent());
    obbm2 = makeOBBModel(secondCollision,scn,getExtent());

    obbm1->setSelfCollision(true);
    obbm2->setSelfCollision(true);

    typename BroadPhase::SPtr pbroadphase = sofa::core::objectmodel::New<BroadPhase>();
    BroadPhase & broadphase = *pbroadphase;

    for(int i = 0 ; i < 2 ; ++i){
        if(!GENTest(obbm1.get(),obbm2.get(),broadphase))
            return false;

        randMoving(obbm1.get(),min,max,seed);
        randMoving(obbm2.get(),min,max,seed);
    }

    return true;
}


template <class BroadPhase>
bool BroadPhaseTest<BroadPhase>::randDense(){
    ////*!randTest(i,20,20,Vector3(-5,-5,-5),Vector3(5,5,5))*/
    for(int i = 0 ; i < 100 ; ++i){
        if(/*!randTest(i,2,2,Vector3(-2,-2,-2),Vector3(2,2,2))*/!randTest(i,40,20,sofa::defaulttype::Vector3(-5,-5,-5),sofa::defaulttype::Vector3(5,5,5))){
            //std::cout<<"FAIL seed number "<<i<<std::endl;
            ADD_FAILURE() <<"FAIL seed number "<<i<< std::endl;
            return false;
        }
    }

    return true;
}

template <class BroadPhase>
bool BroadPhaseTest<BroadPhase>::randSparse(){
    for(int i = 0 ; i < 1000 ; ++i){
        if(/*!randTest(i,1,1,Vector3(-2,-2,-2),Vector3(2,2,2))*/!randTest(i,2,1,sofa::defaulttype::Vector3(-5,-5,-5),sofa::defaulttype::Vector3(5,5,5))){
            //std::cout<<"FAIL seed number "<<i<<std::endl;
            ADD_FAILURE() <<"FAIL seed number "<<i<< std::endl;
            return false;
        }
    }

    return true;
}

#endif
