#define SOFA_FLEXIBLE_ImageDensityMass_CPP

#include "ImageDensityMass.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec3Types.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;



SOFA_DECL_CLASS(ImageDensityMass)


// Register in the Factory
int ImageDensityMassClass = core::RegisterObject("Define a global mass matrix including non diagonal terms")
#ifndef SOFA_FLOAT
        .add< ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction3d,Mat3x3d> >( true )
        .add< ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction2d,Mat3x3d> >()
        .add< ImageDensityMass<Affine3dTypes,core::behavior::ShapeFunction3d,Affine3dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction3f,Mat3x3f> >()
        .add< ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction2f,Mat3x3f> >()
        .add< ImageDensityMass<Affine3fTypes,core::behavior::ShapeFunction3f,Affine3fMass> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_Flexible_API ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction3d,Mat3x3d>;
template class SOFA_Flexible_API ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction2d,Mat3x3d>;
template class SOFA_Flexible_API ImageDensityMass<Affine3dTypes,core::behavior::ShapeFunction3d,Affine3dMass>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_Flexible_API ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction3f,Mat3x3f>;
template class SOFA_Flexible_API ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction2f,Mat3x3f>;
template class SOFA_Flexible_API ImageDensityMass<Affine3fTypes,core::behavior::ShapeFunction3f,Affine3fMass>;
#endif

} // namespace mass

} // namespace component

} // namespace sofa

