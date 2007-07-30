#ifndef SOFA_GPU_CUDA_CUDARIGIDMAPPING_H
#define SOFA_GPU_CUDA_CUDARIGIDMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
class RigidMappingInternalData<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::CudaVec3fTypes::VecDeriv tmp;
};

template <>
void RigidMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void RigidMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaRigid3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
