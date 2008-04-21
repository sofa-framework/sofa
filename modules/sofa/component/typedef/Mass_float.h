#ifndef SOFA_TYPEDEF_MASS_FLOAT_H
#define SOFA_TYPEDEF_MASS_FLOAT_H

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Mat.h>

//Typedef to easily use mass with float type
#include <sofa/component/mass/DiagonalMass.h>
#include <sofa/component/mass/MatrixMass.h>
#include <sofa/component/mass/UniformMass.h>

//Diagonal Mass
//---------------------
//Deformable
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Vec1fTypes,float> DiagonalMass1f;
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Vec2fTypes,float> DiagonalMass2f;
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Vec3fTypes,float> DiagonalMass3f;
//---------------------
//Rigid
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Rigid2fTypes,sofa::defaulttype::Rigid2fMass> DiagonalMassRigid2f;
typedef sofa::component::mass::DiagonalMass<sofa::defaulttype::Rigid3fTypes,sofa::defaulttype::Rigid3fMass> DiagonalMassRigid3f;



//Matrix Mass
//---------------------
//Deformable
typedef sofa::component::mass::MatrixMass<sofa::defaulttype::Vec2fTypes, sofa::defaulttype::Mat2x2f> MatrixMass2f;
typedef sofa::component::mass::MatrixMass<sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Mat3x3f> MatrixMass3f;
//---------------------
//Rigid
//Not defined yet



//Uniform Mass float
//---------------------
//Deformable
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec1fTypes,float> UniformMass1f;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec2fTypes,float> UniformMass2f;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec3fTypes,float> UniformMass3f;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Vec6fTypes,float> UniformMass6f;
//---------------------
//Rigid
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Rigid2fTypes,sofa::defaulttype::Rigid2fMass> UniformMassRigid2f;
typedef sofa::component::mass::UniformMass<sofa::defaulttype::Rigid3fTypes,sofa::defaulttype::Rigid3fMass> UniformMassRigid3f;
//Not defined for 1D, and 6D


#ifdef SOFA_FLOAT

typedef DiagonalMass1f        DiagonalMass1;
typedef DiagonalMass2f 	      DiagonalMass2;
typedef DiagonalMass3f 	      DiagonalMass3;
typedef DiagonalMassRigid2f   DiagonalMassRigid2;
typedef DiagonalMassRigid3f   DiagonalMassRigid3;
typedef MatrixMass2f 	      MatrixMass2;
typedef MatrixMass3f 	      MatrixMass3;
typedef UniformMass1f 	      UniformMass1;
typedef UniformMass2f 	      UniformMass2;
typedef UniformMass3f 	      UniformMass3;
typedef UniformMass6f 	      UniformMass6;
typedef UniformMassRigid2f    UniformMassRigid2;
typedef UniformMassRigid3f    UniformMassRigid3;
#endif

#endif
