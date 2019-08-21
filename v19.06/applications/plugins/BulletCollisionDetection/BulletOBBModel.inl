#include "BulletOBBModel.h"



namespace sofa
{

namespace component
{

namespace collision
{

template <class DataTypes>
TBulletOBBModel<DataTypes>::TBulletOBBModel()
    : TOBBModel<DataTypes>()
    , margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    , _bt_cshape(0x0)
{}

template<class DataTypes>
TBulletOBBModel<DataTypes>::TBulletOBBModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : TOBBModel<DataTypes>(_mstate)
    , margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    , _bt_cshape(0x0)
{}

static btRigidBody* localCreateRigidBody(float mass, const btTransform& startTransform,btCollisionShape* shape,float processingThreshold)
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
    body->setContactProcessingThreshold(processingThreshold);
	

    return body;
}

template <class DataTypes>
void TBulletOBBModel<DataTypes>::initBullet(){
    sofa::core::objectmodel::BaseObject::f_listening.setValue(true);
    _bt_cshape = new btCompoundShape();
  //  _bt_cshape->setMargin(margin.getValue());

    btTransform trans;
    btVector3 bex;
    btVector3 bc;
    btQuaternion bquat;
    SReal marginValue = margin.getValue();
    for(int i = 0 ; i < this->size ; ++i){
        const Coord & ex = this->extents(i);
        const Coord & c = this->center(i);
        const Quaternion quat = this->orientation(i);
        bex.setValue(ex[0] + marginValue,ex[1] + marginValue,ex[2] + marginValue);
       // bex.setValue(ex[0],ex[1],ex[2]);
        bc.setValue(c[0],c[1],c[2]);
        bquat.setValue(quat[0],quat[1],quat[2],quat[3]);

        trans.setOrigin(bc);
        trans.setRotation(bquat);

        btBoxShape * obb = new btBoxShape(bex);
       // obb->setMargin(marginValue);
        //obb->setMargin(0);
        obb->setUserPointer((void*)(size_t)i);

        _garbage.push(obb);
        _bt_cshape->addChildShape(trans,obb);
    }

    //to be commented
    //_bt_cshape->setMargin(margin.getValue());

    btTransform startTransform;
    startTransform.setIdentity();

    _bt_collision_object = localCreateRigidBody(1,startTransform,_bt_cshape,margin.getValue());///PROCESSING THRESHOLD ??? CONTINUE HERE MORENO !!!!
}


template <class DataTypes>
void TBulletOBBModel<DataTypes>::updateBullet(){
    btTransform trans;
    btVector3 bc;
    btQuaternion bquat;
    for(int i = 0 ; i < this->size ; ++i){
        const Coord & c = this->center(i);
        const Quaternion quat = this->orientation(i);
        bc.setValue(c[0],c[1],c[2]);
        bquat.setValue(quat[0],quat[1],quat[2],quat[3]);

        trans.setOrigin(bc);
        trans.setRotation(bquat);
        _bt_cshape->updateChildTransform(i,trans,false);
    }
}


template <class DataTypes>
void TBulletOBBModel<DataTypes>::cleanGarbage(){
    while(!_garbage.empty()){
        delete _garbage.top();
        _garbage.pop();
    }
}

template <class DataTypes>
TBulletOBBModel<DataTypes>::~TBulletOBBModel(){
    cleanGarbage();
}

template <class DataTypes>
void TBulletOBBModel<DataTypes>::init(){
    TOBBModel<DataTypes>::init();
    initBullet();
}


template <class DataTypes>
void TBulletOBBModel<DataTypes>::reinit(){
    delete this->_bt_collision_object;
    delete _bt_cshape;

    cleanGarbage();

    init();
}

template <class DataTypes>
void TBulletOBBModel<DataTypes>::handleEvent(sofa::core::objectmodel::Event * ev){
    if(dynamic_cast<sofa::simulation::CollisionBeginEvent*>(ev)){
        updateBullet();
    }
}

}
}
}
