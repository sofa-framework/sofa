#ifndef SOFA_TYPEDEF_MECHANICAL_FLOAT_H
#define SOFA_TYPEDEF_MECHANICAL_FLOAT_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>

#include <sofa/component/MechanicalObject.h>


//Mechanical Object
//---------------------
//Deformable
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec1fTypes> MechanicalObject1f;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec2fTypes> MechanicalObject2f;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec3fTypes> MechanicalObject3f;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Vec6fTypes> MechanicalObject6f;
//---------------------
//Rigid
typedef sofa::component::MechanicalObject<sofa::defaulttype::Rigid2fTypes> MechanicalObjectRigid2f;
typedef sofa::component::MechanicalObject<sofa::defaulttype::Rigid3fTypes> MechanicalObjectRigid3f;
//---------------------
//Laparoscopic
//Not defined for float

#endif
