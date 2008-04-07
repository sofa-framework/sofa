#ifndef SOFA_TYPEDEF_MECHANICAL_FLOAT_H
#define SOFA_TYPEDEF_MECHANICAL_FLOAT_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>

#include <sofa/component/MechanicalObject.h>

typedef sofa::defaulttype::Vec1fTypes   Particles1f;
typedef Particles1f::Deriv              Deriv1f;
typedef Particles1f::Coord              Coord1f;
typedef sofa::defaulttype::Vec2fTypes   Particles2f;
typedef Particles2f::Deriv              Deriv2f;
typedef Particles2f::Coord              Coord2f;
typedef sofa::defaulttype::Vec3fTypes   Particles3f;
typedef Particles3f::Deriv              Deriv3f;
typedef Particles3f::Coord              Coord3f;
typedef sofa::defaulttype::Vec6fTypes   Particles6f;
typedef Particles6f::Deriv              Deriv6f;
typedef Particles6f::Coord              Coord6f;

typedef sofa::defaulttype::Rigid2fTypes Rigid2f;
typedef Rigid2f::Deriv              DerivRigid2f;
typedef Rigid2f::Coord              CoordRigid2f;
typedef sofa::defaulttype::Rigid3fTypes Rigid3f;
typedef Rigid3f::Quat               Quat3f;
typedef Rigid3f::Quat               DerivRigid3f;
typedef Rigid3f::Coord              CoordRigid3f;

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
