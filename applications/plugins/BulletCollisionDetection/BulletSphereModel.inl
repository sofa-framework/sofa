#include "BulletSphereModel.h"



namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
TBulletSphereModel<DataTypes>::TBulletSphereModel()
    : TSphereModel<DataTypes>()
    , margin(initData(&margin, (SReal)0.05, "margin","Margin used for collision detection within bullet"))
    , _bt_cshape(0x0)
{}

template<class DataTypes>
TBulletSphereModel<DataTypes>::TBulletSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : TSphereModel<DataTypes>(_mstate)
    , margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    , _bt_cshape(0x0)
{}

static btRigidBody* localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape,float /*processingThreshold*/)
{
    btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

    //rigidbody is dynamic if and only if mass is non zero, otherwise static
//    bool isDynamic = (mass != 0.f);

    btVector3 localInertia(0,0,0);
//    if (isDynamic)
//        shape->calculateLocalInertia(mass,localInertia);

    //using motionstate is recommended, it provides interpolation capabilities, and only synchronizes 'active' objects

    btRigidBody* body = new btRigidBody(mass,0,shape,localInertia);

    body->setWorldTransform(startTransform);
   // body->setContactProcessingThreshold(processingThreshold);
	//	 body->setContactProcessingThreshold(0.04f);

    return body;
}

template <class DataTypes>
void TBulletSphereModel<DataTypes>::initBullet(){
    sofa::core::objectmodel::BaseObject::f_listening.setValue(true);
    _bt_cshape = new btCompoundShape();

    const VecCoord & pos = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    int npoints = pos.size();

    const VecReal & radii = this->radius.getValue();

    for(int i = 0 ; i < npoints ; ++i){
        btVector3 btP(pos[i][0],pos[i][1],pos[i][2]);
        btSphereShape * sphere = new btSphereShape(radii[i] + margin.getValue());
        sphere->setUserPointer((void*)(size_t)i);
        _garbage.push(sphere);
        _bt_cshape->addChildShape(btTransform(btQuaternion(0,0,0,1),btP),sphere);
    }

   // _bt_cshape->setMargin(margin.getValue());

    btTransform startTransform;
    startTransform.setIdentity();

    _bt_collision_object = localCreateRigidBody(1,startTransform,_bt_cshape,margin.getValue());///PROCESSING THRESHOLD ??? CONTINUE HERE MORENO !!!!
}


template <class DataTypes>
void TBulletSphereModel<DataTypes>::updateBullet(){
    sofa::core::objectmodel::BaseObject::f_listening.setValue(true);

    const VecCoord & pos = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    int npoints = pos.size();

    btTransform trans;
    trans.setIdentity();
    for(int i = 0 ; i < npoints ; ++i){
        btVector3 btP(pos[i][0],pos[i][1],pos[i][2]);
        btTransform & trans = _bt_cshape->getChildTransform(i);
        trans.setOrigin(btP);
        _bt_cshape->updateChildTransform(i,trans,false);
    }
}


template <class DataTypes>
void TBulletSphereModel<DataTypes>::cleanGarbage(){
    while(!_garbage.empty()){
        delete _garbage.top();
        _garbage.pop();
    }
}

template <class DataTypes>
TBulletSphereModel<DataTypes>::~TBulletSphereModel(){
    cleanGarbage();
}

template <class DataTypes>
void TBulletSphereModel<DataTypes>::init(){
    TSphereModel<DataTypes>::init();
    initBullet();
}


template <class DataTypes>
void TBulletSphereModel<DataTypes>::reinit(){
    delete this->_bt_collision_object;
    delete _bt_cshape;

    cleanGarbage();

    init();
}

template <class DataTypes>
void TBulletSphereModel<DataTypes>::handleEvent(sofa::core::objectmodel::Event * ev){
    if(dynamic_cast<sofa::simulation::CollisionBeginEvent*>(ev)){
        updateBullet();
    }
}

}
}
}
