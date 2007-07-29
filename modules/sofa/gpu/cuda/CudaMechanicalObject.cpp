#include "CudaTypes.h"
#include "CudaMechanicalObject.inl"
#include <sofa/component/collision/SphereModel.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/MappedObject.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaMechanicalObject)

int MechanicalObjectCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::MechanicalObject<CudaVec3fTypes> >()
        .add< component::MechanicalObject<CudaRigid3fTypes> >()
        ;

int CudaSphereModelClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::collision::TSphereModel<CudaVec3fTypes> >()
        ;

int MappedObjectCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::MappedObject<CudaVec3fTypes> >()
        .add< component::MappedObject<CudaRigid3fTypes> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
