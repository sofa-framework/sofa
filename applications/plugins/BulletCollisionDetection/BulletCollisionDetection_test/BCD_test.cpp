#include <SofaTest/BroadPhase_test.h>
#include <SofaTest/PrimitiveCreation.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include "BulletCollisionDetection.h"
#include <SofaTest/PrimitiveCreation.h>
#include <SofaBaseCollision/DefaultPipeline.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include <sofa/helper/RandomGenerator.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/CollisionEndEvent.h>
#include <sofa/simulation/common/CollisionBeginEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>



typedef sofa::defaulttype::Vector3 Vec3;

using namespace sofa::PrimitiveCreationTest;
using sofa::core::objectmodel::New;
using sofa::core::objectmodel::Data;

struct BCD_test : public ::testing::Test{

    template <class ColModel,class CopyConstructor>
    static bool randTest(int seed, int nb_move);

    static bool randTestBulletOBB(int seed,int nb_move);

    static bool randTestBulletConvexHull(int seed,int nb_move);


    static bool trueTest(int seed);

};

static Vec3 st_pos_min(-10,-10,-10);
static Vec3 st_pos_max(10,10,10);
static Vec3 min_extent(0.5,0.5,0.5);
static Vec3 max_extent(3,3,3);

sofa::component::collision::BulletOBBModel::SPtr makeBulletOBB(const Vec3 & p,const double *angles,const int *order,const Vec3 &v,const Vec3 &extents, sofa::simulation::Node::SPtr &father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr obb = father->createChild("obb");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    obbDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //we create a frame that we will rotate like it is specified by the parameters angles and order
    Vec3 x(1,0,0);
    Vec3 y(0,1,0);
    Vec3 z(0,0,1);

    //creating an array of functions which are the rotation so as to perform the rotations in a for loop
    typedef void (*rot)(double,Vec3&,Vec3&,Vec3&);
    rot rotations[3];
    rotations[0] = &rotx;
    rotations[1] = &roty;
    rotations[2] = &rotz;

    //performing the rotations of the frame x,y,z
    for(int i = 0 ; i < 3 ; ++i)
        (*rotations[order[i]])(angles[order[i]],x,y,z);


    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = sofa::defaulttype::Rigid3Types::Coord(p, sofa::defaulttype::Quaternion::createQuaterFromFrame(x,y,z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *obbDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    obb->addObject(obbDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::BulletOBBModel::SPtr obbCollisionModel = New<sofa::component::collision::BulletOBBModel >();
    obb->addObject(obbCollisionModel);

    //editting the OBBModel
    //obbCollisionModel->init();
    obbCollisionModel->resize(1);

    Data<sofa::component::collision::BulletOBBModel::VecCoord> & dVecCoord = obbCollisionModel->writeExtents();
    sofa::component::collision::BulletOBBModel::VecCoord & vecCoord = *(dVecCoord.beginEdit());

    vecCoord[0] = extents;

    dVecCoord.endEdit();

    return obbCollisionModel;
}

static void randTrans(Vec3 & angles,Vec3 & new_pos,int seed){
    new_pos = randVect(st_pos_min,st_pos_max,seed);

    sofa::helper::RandomGenerator randomGenerator;
    randomGenerator.initSeed(seed);
    for(int i = 0 ; i < 3 ; ++i){
        angles[i] = (randomGenerator.random<SReal>())/RAND_MAX *  2 * M_PI;
    }
}

static void transMechaRigid(const Vec3 & angles,const Vec3 & new_pos,sofa::simulation::Node::SPtr & node){
    MechanicalObjectRigid3* mecha = node->get<MechanicalObjectRigid3>(sofa::simulation::Node::SearchDown);

    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *mecha->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    sofa::defaulttype::Quat & quat = positions[0].getOrientation();
    Vec3 & pos  = positions[0].getCenter();

    quat.rotate(angles);
    pos = new_pos;

    dpositions.endEdit();
}

struct copyBulletOBB{
    sofa::component::collision::BulletOBBModel::SPtr operator()(const sofa::component::collision::OBBModel::SPtr & obb_read,sofa::simulation::Node::SPtr &father){
        sofa::simulation::Node::SPtr obb = father->createChild("obb");

        //creating a mechanical object which will be attached to the OBBModel
        MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

        //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
        obbDOF->resize(1);
        Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
        MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

        const MechanicalObjectRigid3 * read_mec = obb_read->getContext()->get<MechanicalObjectRigid3>();
        const Data<MechanicalObjectRigid3::VecCoord> & read_positions = *read_mec->read( sofa::core::VecId::position() );

        //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
        positions[0] = (read_positions.getValue())[0];

        dpositions.endEdit();

        obb->addObject(obbDOF);

        //creating an OBBModel and attaching it to the same node than obbDOF
        sofa::component::collision::BulletOBBModel::SPtr obbCollisionModel = New<sofa::component::collision::BulletOBBModel >();
        obb->addObject(obbCollisionModel);

        //editting the OBBModel
        //obbCollisionModel->init();
        obbCollisionModel->resize(1);


        sofa::component::collision::BulletOBBModel * ptr_obb = obbCollisionModel.get();
        ptr_obb->setName(obb_read->getName());
        sofa::component::collision::BulletOBBModel::VecCoord & extents = *(ptr_obb->ext.beginEdit());

        extents[0] = obb_read->extents(0);

        ptr_obb->ext.endEdit();

        return obbCollisionModel;
    }
};

struct copyBulletConvexHull{
    sofa::component::collision::BulletConvexHullModel::SPtr operator()(const sofa::component::collision::OBBModel::SPtr & obb_read,sofa::simulation::Node::SPtr &father){
        sofa::simulation::Node::SPtr obb = father->createChild("obb");

        //creating a mechanical object which will be attached to the OBBModel
        MechanicalObjectRigid3::SPtr cv_hullDOF = New<MechanicalObjectRigid3>();

        //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
        cv_hullDOF->resize(1);
        Data<MechanicalObjectRigid3::VecCoord> & dpositions = *cv_hullDOF->write( sofa::core::VecId::position() );
        MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

        const MechanicalObjectRigid3 * read_mec = obb_read->getContext()->get<MechanicalObjectRigid3>();
        const Data<MechanicalObjectRigid3::VecCoord> & read_positions = *read_mec->read( sofa::core::VecId::position() );

        //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
        positions[0] = (read_positions.getValue())[0];

        dpositions.endEdit();

        obb->addObject(cv_hullDOF);

        //creating an OBBModel and attaching it to the same node than obbDOF
        sofa::component::collision::BulletConvexHullModel::SPtr cv_hull_model = New<sofa::component::collision::BulletConvexHullModel >();
        obb->addObject(cv_hull_model);

        //editting the OBBModel
        //obbCollisionModel->init();
        cv_hull_model->resize(1);


        sofa::component::collision::BulletConvexHullModel * ptr_cv_hull = cv_hull_model.get();
        ptr_cv_hull->setName(obb_read->getName());
//        sofa::component::collision::BulletOBBModel::VecCoord & extents = *(ptr_obb->ext.beginEdit());

//        extents[0] = obb_read->extents(0);

        sofa::component::collision::OBBModel::VecCoord vs;
        obb_read->vertices(0,vs);

        sofa::component::collision::BulletConvexHullModel::VecCoord & ch_points = *(cv_hull_model->CHPoints.beginEdit());
        for(unsigned int i = 0 ; i < vs.size() ; ++i)
            ch_points.push_back(obb_read->localCoordinates(vs[i],0));

        cv_hull_model->CHPoints.endEdit();

        cv_hull_model->positionDefined.setValue(true);
        //cv_hull_model->computeConvexHullDecomposition.setValue(true);

        return cv_hull_model;
    }
};


//static sofa::component::collision::OBBModel::SPtr copyOBB(const sofa::component::collision::BulletOBBModel::SPtr & obb_read,sofa::simulation::Node::SPtr &father){
//    sofa::simulation::Node::SPtr obb = father->createChild("obb");

//    //creating a mechanical object which will be attached to the OBBModel
//    MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

//    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
//    obbDOF->resize(1);
//    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
//    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

//    const MechanicalObjectRigid3 * read_mec = obb_read->getContext()->get<MechanicalObjectRigid3>();
//    const Data<MechanicalObjectRigid3::VecCoord> & read_positions = *read_mec->read( sofa::core::VecId::position() );

//    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
//    positions[0] = (read_positions.getValue())[0];

//    dpositions.endEdit();

//    obb->addObject(obbDOF);

//    //creating an OBBModel and attaching it to the same node than obbDOF
//    sofa::component::collision::OBBModel::SPtr obbCollisionModel = New<sofa::component::collision::OBBModel >();
//    obb->addObject(obbCollisionModel);

//    //editting the OBBModel
//    //obbCollisionModel->init();
//    obbCollisionModel->resize(1);


//    sofa::component::collision::OBBModel * ptr_obb = obbCollisionModel.get();
//    ptr_obb->setName(obb_read->getName());
//    sofa::component::collision::OBBModel::VecCoord & extents = *(ptr_obb->ext.beginEdit());

//    extents[0] = obb_read->extents(0);

//    ptr_obb->ext.endEdit();

//    return obbCollisionModel;
//}

static sofa::component::collision::OBBModel::SPtr makeRandOBB(const Vec3 & pos_min,const Vec3 & pos_max,sofa::simulation::Node::SPtr &father,int seed){
    Vec3 p = randVect(pos_min,pos_max,seed);
    SReal angles[3];
    sofa::helper::RandomGenerator randomGenerator;
    randomGenerator.initSeed(seed);

    for(int i = 0 ; i < 3 ; ++i){
        angles[i] = (randomGenerator.random<SReal>())/RAND_MAX *  2 * M_PI;

        //ret[i] = ((double)(rand())/RAND_MAX) * extents[i] + min[i];

    }

    int order[3];order[0] = 0;order[1] = 1;order[2] = 2;


    Vec3 v(0,0,0);
    Vec3 extents(randVect(Vec3(0.5,0.5,0.5),Vec3(3,3,3),seed));

    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(p,angles,order,v,extents,father);

    return obbmodel;
}


//static sofa::component::collision::BulletOBBModel::SPtr makeRandBulletOBB(const Vec3 & pos_min,const Vec3 & pos_max,sofa::simulation::Node::SPtr &father){
//    Vec3 p = randVect(pos_min,pos_max);
//    SReal angles[3];

//    for(int i = 0 ; i < 3 ; ++i){
//        angles[i] = ((SReal)rand())/RAND_MAX *  2 * M_PI;

//        //ret[i] = ((double)(rand())/RAND_MAX) * extents[i] + min[i];

//    }

//    int order[3];order[0] = 0;order[1] = 1;order[2] = 2;


//    Vec3 v(0,0,0);
//    Vec3 extents(randVect(min_extent,max_extent));

//    sofa::component::collision::BulletOBBModel::SPtr obbmodel = makeBulletOBB(p,angles,order,v,extents,father);

//    return obbmodel;
//}

template <class CollModel,class CopyConstructor>
bool BCD_test::randTest(int seed,int nb_move){

    sofa::simulation::Node::SPtr bullet_scn = New<sofa::simulation::tree::GNode>();
    sofa::simulation::Node::SPtr sofa_scn = New<sofa::simulation::tree::GNode>();

    sofa::core::ExecParams * default_params = sofa::core::ExecParams::defaultInstance();

    //elements within the sofa scene
    sofa::component::collision::BruteForceDetection::SPtr bfd = New<sofa::component::collision::BruteForceDetection>();
    sofa::component::collision::DefaultPipeline::SPtr sofa_pipeline = New<sofa::component::collision::DefaultPipeline>();
    sofa::component::collision::NewProximityIntersection::SPtr new_prox = New<sofa::component::collision::NewProximityIntersection>();

    //setting scnene parameters
    new_prox->setContactDistance((SReal)0.5);
    new_prox->setAlarmDistance((SReal)0.5);

    //adding elements to the scene
    sofa_scn->addObject(bfd);
    sofa_scn->addObject(sofa_pipeline);
    sofa_scn->addObject(new_prox);

    //copying collision models and adding them to the scene
    sofa::component::collision::OBBModel::SPtr obb1s = makeRandOBB(st_pos_min,st_pos_max,sofa_scn,seed);
    sofa::component::collision::OBBModel::SPtr obb2s = makeRandOBB(st_pos_min,st_pos_max,sofa_scn,seed);

    //setting id to collision models
    std::string name1("1");
    std::string name2("2");
    obb1s->setName(name1);
    obb2s->setName(name2);

    //initializing elements of the scene
    sofa_scn->init(default_params);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    //elements within the bullet scene
    sofa::component::collision::BulletCollisionDetection::SPtr bcd = New<sofa::component::collision::BulletCollisionDetection>();
    sofa::component::collision::DefaultPipeline::SPtr bullet_pipeline = New<sofa::component::collision::DefaultPipeline>();
    sofa::component::collision::BulletIntersection::SPtr bullet_inter = New<sofa::component::collision::BulletIntersection>();

    //setting scnene parameters
    bullet_inter->setContactDistance((SReal)0.5);

    //adding elements to the scene
    bullet_scn->addObject(bcd);
    bullet_scn->addObject(bullet_pipeline);
    bullet_scn->addObject(bullet_inter);

    //creating collision models and adding them to the scene
    CopyConstructor cbo;
    typename CollModel::SPtr obb1 = cbo(obb1s,bullet_scn);
    typename CollModel::SPtr obb2 = cbo(obb2s,bullet_scn);

    //initializing elements of the scene
    bullet_scn->init(default_params);


    Vec3 angles;
    Vec3 new_pos;
    for(int i = 0 ; i < nb_move ; ++i){
        //processing collision pipeline
        {
            {
                sofa::simulation::CollisionBeginEvent evBegin;
                sofa::simulation::PropagateEventVisitor eventPropagation( default_params, &evBegin);
                eventPropagation.execute(sofa_scn.get());
                eventPropagation.execute(bullet_scn.get());
            }

            sofa::simulation::CollisionVisitor act(default_params);
            //act.setTags(this->getTags());
            act.execute( sofa_scn.get() );
            act.execute( bullet_scn.get() );

            {
                sofa::simulation::CollisionEndEvent evEnd;
                sofa::simulation::PropagateEventVisitor eventPropagation( default_params, &evEnd);
                eventPropagation.execute(sofa_scn.get());
                eventPropagation.execute(bullet_scn.get());
            }
        }

        //////////////////////////////////////////Checking the results//////////////////////////////////////////////////
        //recovering contacts of sofa and bullet
        const sofa::component::collision::BruteForceDetection::DetectionOutputMap & sofa_contacts = bfd->getDetectionOutputs();
        const sofa::component::collision::BruteForceDetection::DetectionOutputMap & bullet_contacts = bcd->getDetectionOutputs();

        if(!(sofa_contacts.size() == 0 && bullet_contacts.size() == 0)){
            sofa::component::collision::BruteForceDetection::DetectionOutputMap::const_iterator it_sofa_contacts = sofa_contacts.begin();
            sofa::component::collision::BulletCollisionDetection::DetectionOutputMap::const_iterator it_bullet_contacts = bullet_contacts.begin();

            for(;it_sofa_contacts != sofa_contacts.end() ; ++it_sofa_contacts){
                if((*it_sofa_contacts).first.first != (*it_sofa_contacts).first.second){//the only possible collision
                    if((*it_sofa_contacts).second->size() > 0){//there is a real contact
                        for(;it_bullet_contacts != bullet_contacts.end() ; ++it_bullet_contacts){//searching it in the bullet pipeline
                            if((it_bullet_contacts->second->size() > 0) && ((it_bullet_contacts->first.first->getName() == name1 &&  it_bullet_contacts->first.second->getName() == name2) ||
                                                                            (it_bullet_contacts->first.first->getName() == name2 &&  it_bullet_contacts->first.second->getName() == name1))){
        //                        return false;
                            }
                            else if(it_bullet_contacts->second->size() > 0){//found not existing contact
                                return false;
                            }
                        }
                        //return true;
                    }
                }
            }
        }

        //moving collision models
        randTrans(angles,new_pos,seed);
        transMechaRigid(angles,new_pos,sofa_scn);
        transMechaRigid(angles,new_pos,bullet_scn);
    }

    return true;
}

bool BCD_test::randTestBulletOBB(int seed,int nb_move){
    return randTest<sofa::component::collision::BulletOBBModel,copyBulletOBB>(seed,nb_move);
}

bool BCD_test::randTestBulletConvexHull(int seed,int nb_move){
    return randTest<sofa::component::collision::BulletConvexHullModel,copyBulletConvexHull>(seed,nb_move);
}

TEST_F(BCD_test, bcd_test_bullet ) { ASSERT_TRUE( randTestBulletOBB(0,100)); }
TEST_F(BCD_test, bcd_test ) { ASSERT_TRUE( randTestBulletConvexHull(0,100)); }

