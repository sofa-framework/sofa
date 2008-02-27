#include "CudaTypes.h"
#include "CudaFixedConstraint.inl"
#include <sofa/component/constraint/BoxConstraint.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaFixedConstraint)

int FixedConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::constraint::FixedConstraint<CudaVec3fTypes> >()
        .add< component::constraint::FixedConstraint<CudaVec3f1Types> >()
        ;

int BoxConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::constraint::BoxConstraint<CudaVec3fTypes> >()
        .add< component::constraint::BoxConstraint<CudaVec3f1Types> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
