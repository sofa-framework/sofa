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
        .add< ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction3d> >( true )
        .add< ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction2d> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction3f> >()
        .add< ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction2f> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction3d>;
template class ImageDensityMass<Vec3dTypes,core::behavior::ShapeFunction2d>;
#endif
#ifndef SOFA_DOUBLE
template class ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction3f>;
template class ImageDensityMass<Vec3fTypes,core::behavior::ShapeFunction2f>;
#endif

} // namespace mass

} // namespace component

} // namespace sofa

