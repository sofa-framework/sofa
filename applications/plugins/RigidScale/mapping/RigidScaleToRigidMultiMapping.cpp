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

int RigidScaleToRigidMultiMappingClass = core::RegisterObject("Convert a rigid in addition to a scale into a rigid, this mapping is designed to work with affine articulated systems constraint by rigid and scale.")
.add< RigidScaleToRigidMultiMapping<Rigid3Types, Vec3Types, Rigid3Types> >()

;

template class SOFA_RIGIDSCALE_API RigidScaleToRigidMultiMapping<Rigid3Types, Vec3Types, Rigid3Types>;


}//namespace forcefield
}// namespace component
}//namespace sofa
