#ifndef SOFA_TYPEDEF_MECHANICAL_DOUBLE_H
#define SOFA_TYPEDEF_MECHANICAL_DOUBLE_H


#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>


#include <sofa/component/MechanicalObject.h>


//Mechanical Object
//---------------------
//Deformable
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec1dTypes> MechanicalObject1d;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec2dTypes> MechanicalObject2d;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec3dTypes> MechanicalObject3d;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec6dTypes> MechanicalObject6d;
//---------------------
//Rigid
typedef sofa::component::MechanicalObject<sofa::defaulttype::Rigid2dTypes> MechanicalObjectRigid2d;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Rigid3dTypes> MechanicalObjectRigid3d;
//---------------------
//Laparoscopic
typedef sofa::component::MechanicalObject<sofa::defaulttype::LaparoscopicRigid3Types> MechanicalObjectLaparoscopicRigid3d;

#endif
