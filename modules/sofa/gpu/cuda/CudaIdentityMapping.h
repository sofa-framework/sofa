#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/IdentityMapping.h>
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
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
