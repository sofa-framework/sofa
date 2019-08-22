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

int RigidScaleToAffineMultiMappingClass = core::RegisterObject("Convert a rigid in addition to a scale into an affine without shearing effect.")
.add< RigidScaleToAffineMultiMapping<Rigid3Types, Vec3Types, Affine3dTypes> >()

;

template class SOFA_RIGIDSCALE_API RigidScaleToAffineMultiMapping<Rigid3Types, Vec3Types, Affine3dTypes>;


}//namespace forcefield
}// namespace component
}//namespace sofa
