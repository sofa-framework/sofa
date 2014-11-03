#include "BulletCollisionDetection.h"
#include <sofa/core/ObjectFactory.h>

namespace sofa{namespace component{namespace collision{

//SOFA_DECL_CLASS(BulletCollisionDetection)

int BulletIntersctionClass = core::RegisterObject("Intersection to use with BulletCollisionDetection")
        .add< BulletIntersection >()
        ;

int BulletCollisionDetectionClass = core::RegisterObject("Collision detection using bullet physics pipeline")
        .add< BulletCollisionDetection >()
        ;

BulletCollisionDetection::BulletCollisionDetection()
    : useSimpleBroadPhase(initData(&useSimpleBroadPhase, false, "useSimpleBroadPhase", "enable/disable simple broad phase"))
    , useSAP(initData(&useSAP, false, "useSAP", "enable/disable sweep and prune"))
    , box(initData(&box, "SAP box", "box used if using sweep and prune"))
{
}

BulletCollisionDetection::~BulletCollisionDetection(){
    delete _bt_world;
    delete _bt_broadphase;
    delete _bt_collision_configuration;
    delete _bt_dispatcher;
}


void BulletCollisionDetection::init(){
    //gContactBreakingThreshold = 0;

    _bt_inter_method = this->getContext()->get<BulletIntersection>();

    if(useSimpleBroadPhase.getValue()){
        _bt_broadphase = new btSimpleBroadphase();
    }
    else if(useSAP.getValue()){
        btVector3 min(box.getValue()[0][0],box.getValue()[0][1],box.getValue()[0][2]);
        btVector3 max(box.getValue()[1][0],box.getValue()[1][1],box.getValue()[1][2]);
        _bt_broadphase = new bt32BitAxisSweep3(min,max);
    }
//    else if(useMultiSAP.getValue())
//        _bt_broadphase = new btMultiSapBroadphase();
    else{
        btDbvtBroadphase * broadphase = new btDbvtBroadphase();
        //broadphase->m_deferedcollide = true;
        _bt_broadphase = broadphase;
    }

    _bt_collision_configuration = new btDefaultCollisionConfiguration();
    _bt_dispatcher = new btCollisionDispatcher(_bt_collision_configuration);

    //_bt_world = new btDiscreteDynamicsWorld(_bt_dispatcher,_bt_broadphase,)
    //btGImpactCollisionAlgorithm::registerAlgorithm(_bt_dispatcher);
    static btGImpactCollisionAlgorithm::CreateFunc s_gimpact_cf;
    _bt_dispatcher->registerCollisionCreateFunc(GIMPACT_SHAPE_PROXYTYPE,GIMPACT_SHAPE_PROXYTYPE,&s_gimpact_cf);
    //_bt_dispatcher->registerCollisionCreateFunc(SPHERE_SHAPE_PROXYTYPE,SPHERE_SHAPE_PROXYTYPE,new btSphereSphereCollisionAlgorithm::CreateFunc);
    //btConvexConvexAlgorithm a()

    _bt_world = new btDiscreteDynamicsWorld(_bt_dispatcher,_bt_broadphase,0x0,_bt_collision_configuration);
    _bt_world->getDispatchInfo().m_useContinuous = false;

    _bt_world->setGravity(btVector3(0,0,0));
}


void BulletCollisionDetection::addCollisionModel (core::CollisionModel *cm){
    BulletCollisionModel * btcm = dynamic_cast<BulletCollisionModel*>(cm);
    if(btcm && !(btcm->handled())){
        btcm->setHandled(true);
        btRigidBody * rb = dynamic_cast<btRigidBody*>(btcm->getBtCollisionObject());
        if(rb){
            _bt_world->addRigidBody(rb);
            _bt2sofa_cm[btcm->getBtCollisionObject()] = cm;
        }
        else{
            std::cerr<<"btCollisionObject type not supported in "<<__FILE__<<" line "<<__LINE__<<std::endl;
        }
    }
}


void BulletCollisionDetection::beginNarrowPhase(){
    core::collision::NarrowPhaseDetection::beginNarrowPhase();

    for(int i = 0 ; i < _bt_dispatcher->getNumManifolds() ; ++i){
        btPersistentManifold* contactpair = _bt_dispatcher->getManifoldByIndexInternal(i);
        contactpair->clearManifold();
    }

    _bt_world->performDiscreteCollisionDetection();
    //_bt_world->stepSimulation(this->getContext()->getDt());

    for(int i = 0 ; i < _bt_dispatcher->getNumManifolds() ; ++i){

        btPersistentManifold* contactpair = _bt_dispatcher->getManifoldByIndexInternal(i);

        sofa::core::CollisionModel * finalcm1 = _bt2sofa_cm[contactpair->getBody0()];
        sofa::core::CollisionModel * finalcm2 = _bt2sofa_cm[contactpair->getBody1()];

        bool swapModels;
        const BtInterManager<BulletIntersection>* contactFiller = _bt_inter_method->findIntersectionManager(finalcm1,finalcm2,swapModels);

        if(!contactFiller)
            continue;

        if(swapModels){
            //std::cout<<"SWAPPING !!"<<std::endl;
            std::swap(finalcm1,finalcm2);
        }

        sofa::core::collision::DetectionOutputVector*& outputs = this->getDetectionOutputs(finalcm1, finalcm2);
        contactFiller->fillContacts(finalcm1,finalcm2,*contactpair,*_bt_inter_method,outputs,swapModels);

//        core::collision::ElementIntersector* intersector = _bt_inter_method.findIntersector(finalcm1, finalcm2, swapModels);
//        intersector->beginIntersect(finalcm1, finalcm2, outputs);//creates outputs if null

//        for(int j = 0 ; j < contactpair->getNumContacts() ; ++j){
//            const btManifoldPoint& manpt = contactpair->getContactPoint(i);

//            //outputs->resize(outputs->size() + 1);
//            //output
//        }
    }
}

void BulletCollisionDetection::draw(const core::visual::VisualParams* /*vparams*/)
{
//    if (!bDraw.getValue()) return;

//        /*const*/ DetectionOutputMap & outputsMap = const_cast<DetectionOutputMap &>(this->getDetectionOutputs());

//        glDisable(GL_LIGHTING);
//        glColor3f(1.0, 0.0, 1.0);
//        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
//        glLineWidth(3);
//        glPointSize(5);

//        for (DetectionOutputMap::iterator it = outputsMap.begin(); it!=outputsMap.end(); it++)
//        {
//            sofa::helper::vector<sofa::core::collision::DetectionOutput>& outputs_ = *(dynamic_cast<sofa::helper::vector<sofa::core::collision::DetectionOutput>* >(it->second));
//            for (int i = 0 ; i < outputs_.size() ; ++i)
//            {
//                outputs_[i].elem.first.draw(vparams);
//                outputs_[i].elem.second.draw(vparams);
//            }
//        }
//        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//        glLineWidth(1);
//        glPointSize(1);

}

}}}
