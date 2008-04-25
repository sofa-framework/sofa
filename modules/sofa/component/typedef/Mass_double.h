#ifndef SOFA_TYPEDEF_MASS_DOUBLE_H
#define SOFA_TYPEDEF_MASS_DOUBLE_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Mat.h>

//Typedef to easily use mass with double type
#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/component/mass/MatrixMass.h>
#include <sofa/component/mass/UniformMass.h>

//Diagonal Mass
//---------------------
//Deformable
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Vec1dTypes,double> DiagonalMass1d;
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Vec2dTypes,double> DiagonalMass2d;
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Vec3dTypes,double> DiagonalMass3d;
//---------------------
//Rigid
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Rigid2dTypes,sofa::defaulttype::Rigid2dMass> DiagonalMassRigid2d;
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Rigid3dTypes,sofa::defaulttype::Rigid3dMass> DiagonalMassRigid3d;



//Matrix Mass
//---------------------
//Deformable
typedef sofa::component::mass::MatrixMass<sofa::defaulttype::Vec2dTypes, sofa::defaulttype::Mat2x2d> MatrixMass2d;
typedef sofa::component::mass::MatrixMass<sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Mat3x3d> MatrixMass3d;
//---------------------
//Rigid
//Not defined yet



//Uniform Mass double
//---------------------
//Deformable
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec1dTypes,double> UniformMass1d;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec2dTypes,double> UniformMass2d;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec3dTypes,double> UniformMass3d;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec6dTypes,double> UniformMass6d;
//---------------------
//Rigid
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Rigid2dTypes,sofa::defaulttype::Rigid2dMass> UniformMassRigid2d;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Rigid3dTypes,sofa::defaulttype::Rigid3dMass> UniformMassRigid3d;
//Not defined for 1D, and 6D


#ifndef SOFA_FLOAT

typedef DiagonalMass1d        DiagonalMass1;
typedef DiagonalMass2d 	      DiagonalMass2;
typedef DiagonalMass3d 	      DiagonalMass3;
typedef DiagonalMassRigid2d   DiagonalMassRigid2;
typedef DiagonalMassRigid3d   DiagonalMassRigid3;
typedef MatrixMass2d 	      MatrixMass2;
typedef MatrixMass3d 	      MatrixMass3;
typedef UniformMass1d 	      UniformMass1;
typedef UniformMass2d 	      UniformMass2;
typedef UniformMass3d 	      UniformMass3;
typedef UniformMass6d 	      UniformMass6;
typedef UniformMassRigid2d    UniformMassRigid2;
typedef UniformMassRigid3d    UniformMassRigid3;
#endif

#endif
