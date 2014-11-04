#include "BulletCylinderModel.h"


namespace sofa
{

namespace component
{

namespace collision
{

template <class TDataTypes>
TBulletCylinderModel<TDataTypes>::TBulletCylinderModel()
    : TCylinderModel<TDataTypes>()
    , margin(initData(&margin, (SReal)0.05, "margin","Margin used for collision detection within bullet"))
    ,_bt_cshape(0x0)
{}

template<class DataTypes>
TBulletCylinderModel<DataTypes>::TBulletCylinderModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : TCylinderModel<DataTypes>(_mstate)
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

template <class TDataTypes>
void TBulletCylinderModel<TDataTypes>::initBullet(){

	 sofa::core::objectmodel::BaseObject::f_listening.setValue(true);
	 _bt_cshape = new btCompoundShape();

	 btTransform trans;
	 btQuaternion quat;
	 btVector3 bt_center;

	 btCylinderShape* bt_cylinder;

	for(int i=0; i < this->size ; i++){
        if(this->local_axis(i)==typename DataTypes::Vec3(0.0, 0.0, 1.0)){
            bt_cylinder = new btCylinderShapeZ(btVector3(this->radius(i), this->radius(i), this->height(i)));
		}
        else if(this->local_axis(i)==typename DataTypes::Vec3(0.0, 1.0, 0.0)){
            bt_cylinder = new btCylinderShape(btVector3(this->radius(i),  this->height(i), this->radius(i)));
		}
        else if(this->local_axis(i)==typename DataTypes::Vec3(1.0, 0.0, 0.0)){
            bt_cylinder = new btCylinderShapeX(btVector3(this->height(i), this->radius(i), this->radius(i)));
		}
		else {
			assert(0 && "wrong local axis specified for cylinder");
			continue;
		}

        bt_cylinder->setMargin(margin.getValue());

		quat.setValue((this->orientation(i))[0],(this->orientation(i))[1],(this->orientation(i))[2],(this->orientation(i))[3]);
		bt_center.setValue((this->center(i))[0],(this->center(i))[1],(this->center(i))[2]);

		trans.setOrigin(bt_center);
		trans.setRotation(quat);

		_garbage.push(bt_cylinder);
		_bt_cshape->addChildShape(trans, bt_cylinder);
	}

	 _bt_cshape->setMargin(margin.getValue());

	 trans.setIdentity();

	 _bt_collision_object = localCreateRigidBody(1,trans,_bt_cshape,margin.getValue());
}


template <class TDataTypes>
void TBulletCylinderModel<TDataTypes>::updateBullet(){

    btTransform trans;
    btQuaternion quat;
    btVector3 bt_center;

    for(int i = 0 ; i < this->size ; ++i){
        quat.setValue((this->orientation(i))[0],(this->orientation(i))[1],(this->orientation(i))[2],(this->orientation(i))[3]);
        bt_center.setValue((this->center(i))[0],(this->center(i))[1],(this->center(i))[2]);

        trans.setOrigin(bt_center);
        trans.setRotation(quat);
		_bt_cshape->updateChildTransform(i,trans,false);//false because done in computeBoundingTree
    }
}


template <class TDataTypes>
void TBulletCylinderModel<TDataTypes>::cleanGarbage(){
    while(!_garbage.empty()){
        delete _garbage.top();
        _garbage.pop();
    }
}

template <class TDataTypes>
TBulletCylinderModel<TDataTypes>::~TBulletCylinderModel(){
    cleanGarbage();
}

template <class DataTypes>
void TBulletCylinderModel<DataTypes>::init(){
    TCylinderModel<DataTypes>::init();
    initBullet();
}


template <class TDataTypes>
void TBulletCylinderModel<TDataTypes>::reinit(){
    delete this->_bt_collision_object;
    delete _bt_cshape;

    cleanGarbage();

    init();
}

template <class TDataTypes>
void TBulletCylinderModel<TDataTypes>::handleEvent(sofa::core::objectmodel::Event * ev){
    if(dynamic_cast<sofa::simulation::CollisionBeginEvent*>(ev)){
        updateBullet();
    }
}












}
}
}
