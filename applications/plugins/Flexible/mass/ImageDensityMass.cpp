#define SOFA_FLEXIBLE_IMAGEDENSITYMASS_CPP

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
int ImageDensityMassClass = core::RegisterObject("Define a specific mass for each dof")
#ifndef SOFA_FLOAT
        .add< ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction3d,Mat3x3d> >( true )
        .add< ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction2d,Mat3x3d> >()
        .add< ImageDensityMass<Affine3dTypes,core::behavior::ShapeFunction3d,Affine3dMass> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction3f,Mat3x3f> >()
        .add< ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction2f,Mat3x3f> >()
        .add< ImageDensityMass<Affine3fTypes,core::behavior::ShapeFunction3d,Affine3fMass> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction3d,Mat3x3d>;
template class ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction2d,Mat3x3d>;
template class ImageDensityMass<Affine3dTypes,core::behavior::ShapeFunction3f,Affine3dMass>;
#endif
#ifndef SOFA_DOUBLE
template class ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction3f,Mat3x3f>;
template class ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction2f,Mat3x3f>;
template class ImageDensityMass<Affine3fTypes,core::behavior::ShapeFunction3f,Affine3fMass>;
#endif

} // namespace mass

} // namespace component

} // namespace sofa

