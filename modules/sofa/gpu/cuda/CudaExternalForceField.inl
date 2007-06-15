
#ifndef SOFA_GPU_CUDA_CUDAEXTERNALFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAEXTERNALFORCEFIELD_INL

#include "CudaExternalForceField.h"
#include <sofa/component/interactionforcefield/ExternalForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void ExternalForceFieldCuda3f_addForce(unsigned int size,void* f, const void* indices,const void *forces );

};

} // namespace cuda

} // namespace gpu



namespace component
{

namespace interactionforcefield
{

using namespace gpu::cuda;

template<>
void ExternalForceField<sofa::gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord&/* p*/, const VecDeriv& /*v*/)
{
    gpu::cuda::CudaVector<unsigned> indices;
    unsigned n=m_indices.getValue().size();
    indices.resize(n);
    for(unsigned i=0; i<m_indices.getValue().size(); i++)
        indices[i]=m_indices.getValue()[i];
    sofa::gpu::cuda::ExternalForceFieldCuda3f_addForce(m_indices.getValue().size(), f.deviceWrite(),indices.deviceRead(),m_forces.getValue().deviceRead());
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
