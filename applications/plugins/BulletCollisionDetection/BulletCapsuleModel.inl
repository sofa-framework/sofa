#include "BulletCapsuleModel.h"

namespace sofa
{

namespace component
{

namespace collision
{

template <class TDataTypes>
TBulletCapsuleModel<TDataTypes>::TBulletCapsuleModel()
    : TCapsuleModel<TDataTypes>()
    , margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    ,_bt_cshape(0x0)
{}

template<class DataTypes>
TBulletCapsuleModel<DataTypes>::TBulletCapsuleModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : TCapsuleModel<DataTypes>(_mstate)
    , margin(initData(&margin, (SReal)0.04, "margin","Margin used for collision detection within bullet"))
    ,_bt_cshape(0x0)
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


template <class Real>
static btCapsuleShape* makeBtCapsule(Real radius,Real height){
    return new BulletSoftCapsule(radius,height);
}

template <class TDataTypes>
void TBulletCapsuleModel<TDataTypes>::initBullet(){
    sofa::core::objectmodel::BaseObject::f_listening.setValue(true);
    _bt_cshape = new btCompoundShape();

    btTransform trans;
    btQuaternion quat;
    btVector3 bt_center;
    btCapsuleShape * capsule;
    SReal marginValue = margin.getValue();

    for(int i = 0 ; i < this->size ; ++i){
        capsule = makeBtCapsule<SReal>(this->radius(i) + marginValue,this->height(i));
        //capsule->setMargin(margin.getValue());
        //capsule->setUserPointer((void*)i);

        quat.setValue((this->orientation(i))[0],(this->orientation(i))[1],(this->orientation(i))[2],(this->orientation(i))[3]);
        bt_center.setValue((this->center(i))[0],(this->center(i))[1],(this->center(i))[2]);

        trans.setOrigin(bt_center);
        trans.setRotation(quat);

        _garbage.push(capsule);
        _bt_cshape->addChildShape(trans,capsule);
    }

    //_bt_cshape->setMargin(margin.getValue());

    trans.setIdentity();

    _bt_collision_object = localCreateRigidBody(1,trans,_bt_cshape,marginValue);
}

template <class TDataTypes>
void updateCapsuleHeight(btCollisionShape * capsule,typename TDataTypes::Real height){
    (static_cast<BulletSoftCapsule*>(capsule))->setHeight(height);
}

#ifndef SOFA_FLOAT
template <>
void updateCapsuleHeight<defaulttype::Rigid3dTypes>(btCollisionShape * ,typename defaulttype::Rigid3dTypes::Real ){}
#endif

#ifndef SOFA_DOUBLE
template <>
void updateCapsuleHeight<defaulttype::Rigid3fTypes>(btCollisionShape * ,typename defaulttype::Rigid3fTypes::Real ){}
#endif

template <class TDataTypes>
void TBulletCapsuleModel<TDataTypes>::updateBullet(){
    btTransform trans;
    btQuaternion quat;
    btVector3 bt_center;

    for(int i = 0 ; i < this->size ; ++i){
        quat.setValue((this->orientation(i))[0],(this->orientation(i))[1],(this->orientation(i))[2],(this->orientation(i))[3]);
        bt_center.setValue((this->center(i))[0],(this->center(i))[1],(this->center(i))[2]);

        trans.setOrigin(bt_center);
        trans.setRotation(quat);

        updateCapsuleHeight<TDataTypes>(_bt_cshape->getChildShape(i),this->height(i));
        _bt_cshape->updateChildTransform(i,trans,false);//false because done in computeBoundingTree
    }
}


template <class TDataTypes>
void TBulletCapsuleModel<TDataTypes>::cleanGarbage(){
    while(!_garbage.empty()){
        delete _garbage.top();
        _garbage.pop();
    }
}

template <class TDataTypes>
TBulletCapsuleModel<TDataTypes>::~TBulletCapsuleModel(){
    cleanGarbage();
}

template <class DataTypes>
void TBulletCapsuleModel<DataTypes>::init(){
    TCapsuleModel<DataTypes>::init();
    initBullet();
}


template <class TDataTypes>
void TBulletCapsuleModel<TDataTypes>::reinit(){
    delete this->_bt_collision_object;
    delete _bt_cshape;

    cleanGarbage();

    init();
}

template <class TDataTypes>
void TBulletCapsuleModel<TDataTypes>::handleEvent(sofa::core::objectmodel::Event * ev){
    if(dynamic_cast<sofa::simulation::CollisionBeginEvent*>(ev)){
        updateBullet();
    }
}

}
}
}
