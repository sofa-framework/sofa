#define SOFA_FLEXIBLE_ImageDensityMass_CPP

#include "ImageDensityMass.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace mass
{

using namespace sofa::defaulttype;

// Register in the Factory
int ImageDensityMassClass = core::RegisterObject("Define a global mass matrix including non diagonal terms")
        .add< ImageDensityMass<Vec3Types,core::behavior::ShapeFunction3d,Mat3x3d> >( true )
        ;
template class SOFA_Flexible_API ImageDensityMass<Vec3Types,core::behavior::ShapeFunction3d,Mat3x3d>;//template class SOFA_Flexible_API ImageDensityMass<Rigid3Types,core::behavior::ShapeFunction3d,Rigid3Mass>;

} // namespace mass

} // namespace component

} // namespace sofa

