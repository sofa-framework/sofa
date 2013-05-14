#include <sofa/component/collision/DirectSAP.h>
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
#ifdef SOFA_HAVE_BGL
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/init.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
#include "../../../applications/tutorials/objectCreator/ObjectCreator.h"

#include <plugins/Flexible/deformationMapping/ExtensionMapping.h>
#include <plugins/Flexible/deformationMapping/DistanceMapping.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/TreeCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/collision/MeshIntTool.h>
#include <sofa/helper/vector.h>
#include <sofa/component/collision/IncrSAP.h>
#include <stdlib.h>


using namespace sofa::component::collision;

struct DirectSAPTest: public ::testing::Test{
    static double getExtent(){return extent;}

    static bool genTest(sofa::core::CollisionModel * cm1,sofa::core::CollisionModel * cm2 = 0x0);
    static bool genOBBTest(std::vector<Vector3> & obb1,std::vector<Vector3> & obb2);

    static bool test1();
    static bool test2();
    static bool test3();
    static bool test4();
    static bool test5();
    static bool test6();

    static bool randSparse();
    static bool randDense();
    static bool randTest3();

    static double extent;

    static bool randTest(int seed,int nb1,int nb2,const Vector3 & min,const Vector3 & max);
};

double DirectSAPTest::extent = 1.2;


struct IncrSAPTest: public ::testing::Test{
    static double getExtent(){return extent;}

    static bool randSparse();
    static bool randDense();
    static bool randTest3();

    static double extent;

    static bool randTest(int seed,int nb1,int nb2,const Vector3 & min,const Vector3 & max);
};

double IncrSAPTest::extent = 1.2;

//intersection method used for the narrow phase
NewProximityIntersection::SPtr proxIntersection = New<NewProximityIntersection>();


//GENERAL FUNCTIONS
template<class Detection>
bool genTest(sofa::core::CollisionModel * cm1,sofa::core::CollisionModel * cm2,Detection & col_detection);

static Vector3 randVect(const Vector3 & min,const Vector3 & max);

template <class SAPBox>
void getSAPBoxes(sofa::core::CollisionModel * cm,std::vector<SAPBox> & sap_boxes){
    CubeModel * cbm = dynamic_cast<CubeModel*>(cm->getFirst());
    assert(cbm != 0x0);

    for(int i = 0 ; i < cbm->getSize() ; ++i)
        sap_boxes.push_back(SAPBox(Cube(cbm,i)));
}

sofa::component::collision::OBBModel::SPtr makeOBBModel(const std::vector<Vector3> & p,sofa::simulation::Node::SPtr &father,double default_extent);

void randMoving(sofa::core::CollisionModel* cm,const Vector3 & min_vect,const Vector3 & max_vect){
    OBBModel * obbm = dynamic_cast<OBBModel*>(cm->getLast());
    MechanicalObjectRigid3* dof = dynamic_cast<MechanicalObjectRigid3*>(obbm->getMechanicalState());

    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *dof->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *dof->write( sofa::core::VecId::velocity() );
    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();


    for(int i = 0 ; i < dof->getSize() ; ++i){
        if(rand() < RAND_MAX/2.0){//make it move !
            velocities[i] = Vector3(1,1,1);//velocity is used only to know if a primitive moves, its direction is not important
            positions[i] = Rigid3dTypes::Coord(randVect(min_vect,max_vect),Quaternion(0,0,0,1));
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
    cm1->setSelfCollision(true);
    cm2->setSelfCollision(true);

    col_detection.setIntersectionMethod(proxIntersection.get());

    col_detection.addCollisionModel(cm1);
    if(cm2 != 0x0)
        col_detection.addCollisionModel(cm2);

    std::vector<typename Detection::SAPBox> boxes;
    std::vector<std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> > brutInter;

    getSAPBoxes(cm1,boxes);
    if(cm2 != 0x0)
        getSAPBoxes(cm2,boxes);

    //cm1 self intersections
    for(unsigned int i = 0 ; i < boxes.size() ; ++i){
        for(unsigned int j = i + 1 ; j < boxes.size() ; ++j){
//            std::cout<<"colliding models "<<boxes[i].cube.getCollisionModel()->getLast()<<" "<<boxes[j].cube.getCollisionModel()->getLast()<<std::endl;
//            std::cout<<"colliding indices "<<boxes[i].cube.getIndex()<<" "<<boxes[j].cube.getIndex()<<std::endl;
//            std::cout<<"min/max vect"<<std::endl;
//            boxes[i].show();
//            boxes[j].show();
            if(boxes[i].overlaps(boxes[j])){
                brutInter.push_back(std::make_pair((sofa::core::CollisionElementIterator)(boxes[i].cube),(sofa::core::CollisionElementIterator)(boxes[j].cube)));
                //std::cout<<"\tCOLLIDING"<<std::endl;
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
    col_detection.addCollisionModel(cm1);
    if(cm2)
        col_detection.addCollisionModel(cm2);

    col_detection.endBroadPhase();
    col_detection.beginNarrowPhase();

    std::vector<std::pair<sofa::core::CollisionElementIterator,sofa::core::CollisionElementIterator> > SAPInter;

    sofa::helper::vector<sofa::core::collision::DetectionOutput> * res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm1,cm1));
    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            SAPInter.push_back(((*res)[i]).elem);


    res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm1,cm2));

    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            SAPInter.push_back(((*res)[i]).elem);

    res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm2,cm1));

    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            SAPInter.push_back(((*res)[i]).elem);

    res = dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput> *>(col_detection.getDetectionOutputs(cm2,cm2));
    if(res != 0x0)
        for(unsigned int i = 0 ; i < res->size() ; ++i)
            SAPInter.push_back(((*res)[i]).elem);

    std::sort(SAPInter.begin(),SAPInter.end(),c);

    col_detection.endNarrowPhase();

    if(brutInter.size() != SAPInter.size())
        return false;

    unsigned int i;
    for(i = 0 ; i < brutInter.size() ; ++i)
        if(!c.same(brutInter[i],SAPInter[i]))
            break;

    if(i < brutInter.size()){
        std::cout<<"BRUT FORCE PAIRS"<<std::endl;
        for(unsigned int j = 0 ; j < brutInter.size() ; ++j){
            std::cout<<brutInter[j].first.getCollisionModel()->getLast()<<" "<<brutInter[j].second.getCollisionModel()->getLast()<<std::endl;
            std::cout<<brutInter[j].first.getIndex()<<" "<<brutInter[j].second.getIndex()<<std::endl;
            std::cout<<"=="<<std::endl;
        }

        std::cout<<"=========SAP PAIRS"<<std::endl;
        for(unsigned int j = 0 ; j < SAPInter.size() ; ++j){
            std::cout<<SAPInter[j].first.getCollisionModel()->getLast()<<" "<<SAPInter[j].second.getCollisionModel()->getLast()<<std::endl;
            std::cout<<SAPInter[j].first.getIndex()<<" "<<SAPInter[j].second.getIndex()<<std::endl;
            std::cout<<"=="<<std::endl;
        }

        return false;
    }

    return true;
}


bool DirectSAPTest::genTest(sofa::core::CollisionModel * cm1,sofa::core::CollisionModel * cm2){
    cm1->setSelfCollision(true);
    cm2->setSelfCollision(true);
    DirectSAP::SPtr psap = New<DirectSAP>();
    DirectSAP & sap = *psap;

    return GENTest(cm1,cm2,sap);
}


sofa::component::collision::OBBModel::SPtr makeOBBModel(const std::vector<Vector3> & p,sofa::simulation::Node::SPtr &father,double default_extent){
    int n = p.size();
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr obb = father->createChild("obb");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    obbDOF->resize(n);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    for(int i = 0 ; i < n ; ++i)
        positions[i] = Rigid3Types::Coord(p[i],Quaternion(0,0,0,1));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *obbDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    for(int i = 0 ; i < n ; ++i)
        velocities[i] = Vector3(0,0,0);
    dvelocities.endEdit();


    obb->addObject(obbDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::OBBModel::SPtr obbCollisionModel = New<sofa::component::collision::OBBModel >();
    obb->addObject(obbCollisionModel);

    //editting the OBBModel
    typename OBBModel::Real & def_ext = *(obbCollisionModel->default_ext.beginEdit());
    def_ext = default_extent;

    obbCollisionModel->default_ext.endEdit();

    obbCollisionModel->init();
//    Data<sofa::component::collision::OBBModel::VecCoord> & dVecCoord = obbCollisionModel->writeExtents();
//    sofa::component::collision::OBBModel::VecCoord & vecCoord = *(dVecCoord.beginEdit());
//dVecCoord.endEdit();
    obbCollisionModel->computeBoundingTree(0);

    return obbCollisionModel;
}

bool DirectSAPTest::genOBBTest(std::vector<Vector3> & obb1,std::vector<Vector3> & obb2){
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    OBBModel::SPtr obbm1,obbm2;
    obbm1 = makeOBBModel(obb1,scn,getExtent());
    obbm2 = makeOBBModel(obb2,scn,getExtent());

    return genTest(obbm1.get(),obbm2.get());
}

bool DirectSAPTest::test1(){
    extent = 1.2;
    std::vector<Vector3> obb1;
    std::vector<Vector3> obb2;

    obb1.push_back(Vector3(0,0,1));
    obb1.push_back(Vector3(4,0,1));
    obb1.push_back(Vector3(8,0,1));

    obb2.push_back(Vector3(0,0,0));
    obb2.push_back(Vector3(4,0,0));
    obb2.push_back(Vector3(8,0,0));

    return genOBBTest(obb1,obb2);
}

bool DirectSAPTest::test2(){
    extent = 1.2;
    std::vector<Vector3> obb1;
    std::vector<Vector3> obb2;

    obb1.push_back(Vector3(0,0,1));
    obb1.push_back(Vector3(2,0,1));
    obb1.push_back(Vector3(4,0,1));

    obb2.push_back(Vector3(0,0,0));
    obb2.push_back(Vector3(2,0,0));
    obb2.push_back(Vector3(4,0,0));

    return genOBBTest(obb1,obb2);
}


Vector3 randVect(const Vector3 & min,const Vector3 & max){
    Vector3 ret;
    Vector3 extents = max - min;

    for(int i = 0 ; i < 3 ; ++i){
        ret[i] = ((double)(rand())/RAND_MAX) * extents[i] + min[i];
    }

    return ret;
}

bool DirectSAPTest::randTest(int seed,int nb1,int nb2, const Vector3 &min, const Vector3 &max){
    srand(seed);

    std::vector<Vector3> firstCollision;
    std::vector<Vector3> secondCollision;

    for(int i = 0 ; i < nb1 ; ++i)
        firstCollision.push_back(randVect(min,max));

    for(int i = 0 ; i < nb2 ; ++i)
        secondCollision.push_back(randVect(min,max));

    return genOBBTest(firstCollision,secondCollision);
}


bool IncrSAPTest::randTest(int seed,int nb1,int nb2,const Vector3 & min,const Vector3 & max){
    srand(seed);

    std::vector<Vector3> firstCollision;
    std::vector<Vector3> secondCollision;

    for(int i = 0 ; i < nb1 ; ++i)
        firstCollision.push_back(randVect(min,max));

    for(int i = 0 ; i < nb2 ; ++i)
        secondCollision.push_back(randVect(min,max));

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    OBBModel::SPtr obbm1,obbm2;
    obbm1 = makeOBBModel(firstCollision,scn,getExtent());
    obbm2 = makeOBBModel(secondCollision,scn,getExtent());

    obbm1->setSelfCollision(true);
    obbm2->setSelfCollision(true);
    IncrSAP::SPtr psap = New<IncrSAP>();
    IncrSAP & sap = *psap;

    if(!GENTest(obbm1.get(),obbm2.get(),sap))
        return false;

    randMoving(obbm1.get(),min,max);
    randMoving(obbm2.get(),min,max);

    if(!GENTest(obbm1.get(),obbm2.get(),sap))
        return false;

    randMoving(obbm1.get(),min,max);
    randMoving(obbm2.get(),min,max);

    return GENTest(obbm1.get(),obbm2.get(),sap);
}


bool IncrSAPTest::randDense(){
    ////*!randTest(i,20,20,Vector3(-5,-5,-5),Vector3(5,5,5))*/
    for(int i = 0 ; i < 100 ; ++i){
        if(/*!randTest(i,2,2,Vector3(-2,-2,-2),Vector3(2,2,2))*/!randTest(i,40,20,Vector3(-5,-5,-5),Vector3(5,5,5))){
            //std::cout<<"FAIL seed number "<<i<<std::endl;
            return false;
        }
    }

    return true;
}

bool IncrSAPTest::randSparse(){
    for(int i = 0 ; i < 100 ; ++i){
        if(!randTest(i,2,2,Vector3(-2,-2,-2),Vector3(2,2,2))/*!randTest(i,1,1,Vector3(-5,-5,-5),Vector3(5,5,5))*/){
            std::cout<<"FAIL seed number "<<i<<std::endl;
            return false;
        }
    }

    return true;
}


bool DirectSAPTest::randDense(){
    ////*!randTest(i,20,20,Vector3(-5,-5,-5),Vector3(5,5,5))*/
    for(int i = 0 ; i < 100 ; ++i){
        if(/*!randTest(i,2,2,Vector3(-2,-2,-2),Vector3(2,2,2))*/!randTest(i,40,20,Vector3(-5,-5,-5),Vector3(5,5,5))){
//            std::cout<<"FAIL seed number "<<i<<std::endl;
            return false;
        }
    }

    return true;
}


bool DirectSAPTest::randSparse(){
    for(int i = 0 ; i < 100 ; ++i){
        if(/*!randTest(i,2,2,Vector3(-2,-2,-2),Vector3(2,2,2))*/!randTest(i,2,2,Vector3(-5,-5,-5),Vector3(5,5,5))){
//            std::cout<<"FAIL seed number "<<i<<std::endl;
            return false;
        }
    }

    return true;
}

TEST_F(DirectSAPTest, test_1 ) { ASSERT_TRUE( test1()); }
TEST_F(DirectSAPTest, test_2 ) { ASSERT_TRUE( test2()); }
TEST_F(DirectSAPTest, rand_dense_test ) { ASSERT_TRUE( randDense()); }
TEST_F(DirectSAPTest, rand_sparse_test ) { ASSERT_TRUE( randSparse()); }
TEST_F(IncrSAPTest, rand_dense_test ) { ASSERT_TRUE( randDense()); }
TEST_F(IncrSAPTest, rand_sparse_test ) { ASSERT_TRUE( randSparse()); }

