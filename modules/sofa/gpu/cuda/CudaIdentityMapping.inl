#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL

#include "CudaIdentityMapping.h"
#include <sofa/component/mapping/IdentityMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    gpu::cuda::MechanicalObjectCudaVec3f_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}


//////// CudaVec3f1

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::componentmodel::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3f1Types>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::cuda::CudaVec3f1Types>, sofa::core::componentmodel::behavior::MappedModel<gpu::cuda::CudaVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
