#include <initRigidScale.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/Vec.h>
#include <Flexible/types/AffineTypes.h>
#include <sofa/core/Multi2Mapping.inl>

#include <RigidScale/mapping/RigidScaleToAffineMultiMapping.inl>

namespace sofa
{
namespace component
{
namespace mapping
{	

using namespace defaulttype;

SOFA_DECL_CLASS(RigidScaleToAffineMultiMapping)

int RigidScaleToAffineMultiMappingClass = core::RegisterObject("Convert a rigid in addition to a scale into an affine without shearing effect.")
#ifndef SOFA_FLOAT
.add< RigidScaleToAffineMultiMapping<Rigid3dTypes, Vec3dTypes, Affine3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< RigidScaleToAffineMultiMapping<Rigid3fTypes, Vec3fTypes, Affine3fTypes > >()
#endif
;

#ifndef SOFA_FLOAT
template class SOFA_RIGIDSCALE_API RigidScaleToAffineMultiMapping<Rigid3dTypes, Vec3dTypes, Affine3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_RIGIDSCALE_API RigidScaleToAffineMultiMapping<Rigid3fTypes, Vec3fTypes, Affine3fTypes>;
#endif

}//namespace forcefield
}// namespace component
}//namespace sofa
