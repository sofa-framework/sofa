#ifndef BULLET_COLLISION_MODEL_H
#define BULLET_COLLISION_MODEL_H

#include "btBulletCollisionCommon.h"
#include <sofa/core/objectmodel/BaseClass.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Data.h>

class BulletCollisionModel /*: public sofa::core::objectmodel::BaseObject */{
public:
    //SOFA_ABSTRACT_CLASS(BulletCollisionModel,sofa::core::objectmodel::BaseObject);
//    typedef BulletCollisionModel MyType;
//    SOFA_ABSTRACT_CLASS_DECL;

    BulletCollisionModel() : _bt_collision_object(0x0),_handled(false){}

    /**
      *Inits bullet collision shapes from the sofa shapes.
      */
    virtual void initBullet() = 0;

    /**
      *Updates at each time step the bullet shapes from sofa shapes.
      */
    virtual void updateBullet() = 0;

    virtual ~BulletCollisionModel(){delete _bt_collision_object;}

    inline btCollisionObject * getBtCollisionObject(){return _bt_collision_object;}

    inline const btCollisionObject * getBtCollisionObject()const{return _bt_collision_object;}

    /**
      *Returns true if the BulletCollisionModeled has been added to the bullet scene.
      */
    inline bool handled()const{return _handled;}

    inline void setHandled(bool h){_handled = h;}

protected:
    btCollisionObject * _bt_collision_object;//the collision object in the bullet scene
    bool _handled;//true if the bullet collision model has been added to the bullet scene
};
#endif
