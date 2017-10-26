#include <initRigidScale.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>

#include <Flexible/types/AffineTypes.h>

#include <sofa/core/Multi2Mapping.inl>

#include <RigidScale/mapping/RigidScaleToRigidMultiMapping.inl>

namespace sofa
{
namespace component
{
namespace mapping
{	

using namespace defaulttype;

SOFA_DECL_CLASS(RigidScaleToRigidMultiMapping)

int RigidScaleToRigidMultiMappingClass = core::RegisterObject("Convert a rigid in addition to a scale into a rigid, this mapping is designed to work with affine articulated systems constraint by rigid and scale.")
#ifndef SOFA_FLOAT
.add< RigidScaleToRigidMultiMapping<Rigid3dTypes, Vec3dTypes, Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidScaleToRigidMultiMapping<Rigid3fTypes, Vec3fTypes, Rigid3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_RIGIDSCALE_API RigidScaleToRigidMultiMapping<Rigid3dTypes, Vec3dTypes, Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_RIGIDSCALE_API RigidScaleToRigidMultiMapping<Rigid3fTypes, Vec3fTypes, Rigid3fTypes>;
#endif

}//namespace forcefield
}// namespace component
}//namespace sofa
