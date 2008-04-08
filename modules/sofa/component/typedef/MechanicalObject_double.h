#ifndef SOFA_TYPEDEF_MECHANICAL_DOUBLE_H
#define SOFA_TYPEDEF_MECHANICAL_DOUBLE_H


#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>


#include <sofa/component/MechanicalObject.h>


typedef sofa::defaulttype::Vec1dTypes   Particles1d;
typedef Particles1d::VecDeriv           VecDeriv1d;
typedef Particles1d::VecCoord           VecCoord1d;
typedef Particles1d::Deriv              Deriv1d;
typedef Particles1d::Coord              Coord1d;
typedef sofa::defaulttype::Vec2dTypes   Particles2d;
typedef Particles2d::VecDeriv           VecDeriv2d;
typedef Particles2d::VecCoord           VecCoord2d;
typedef Particles2d::Deriv              Deriv2d;
typedef Particles2d::Coord              Coord2d;
typedef sofa::defaulttype::Vec3dTypes   Particles3d;
typedef Particles3d::VecDeriv           VecDeriv3d;
typedef Particles3d::VecCoord           VecCoord3d;
typedef Particles3d::Deriv              Deriv3d;
typedef Particles3d::Coord              Coord3d;
typedef sofa::defaulttype::Vec6dTypes   Particles6d;
typedef Particles6d::VecDeriv           VecDeriv6d;
typedef Particles6d::VecCoord           VecCoord6d;
typedef Particles6d::Deriv              Deriv6d;
typedef Particles6d::Coord              Coord6d;

typedef sofa::defaulttype::Rigid2dTypes Rigid2d;
typedef Rigid2d::VecDeriv               VecDerivRigid2d;
typedef Rigid2d::VecCoord               VecCoordRigid2d;
typedef Rigid2d::Deriv                  DerivRigid2d;
typedef Rigid2d::Coord                  CoordRigid2d;
typedef sofa::defaulttype::Rigid3dTypes Rigid3d;
typedef Rigid3d::VecDeriv               VecDerivRigid3d;
typedef Rigid3d::VecCoord               VecCoordRigid3d;
typedef Rigid3d::Quat                   Quat3d;
typedef Rigid3d::Quat                   DerivRigid3d;
typedef Rigid3d::Coord                  CoordRigid3d;

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
