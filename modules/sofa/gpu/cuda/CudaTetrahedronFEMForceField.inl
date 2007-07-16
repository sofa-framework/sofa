#ifndef SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDATETRAHEDRONFEMFORCEFIELD_INL

#include "CudaTetrahedronFEMForceField.h"
#include <sofa/component/forcefield/TetrahedronFEMForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void TetrahedronFEMForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, void* state, const void* velems, void* f, const void* x, const void* v);
    void TetrahedronFEMForceFieldCuda3f_addDForce(unsigned int nbVertex, unsigned int nbElemPerVertex, const void* elems, const void* state, const void* velems, void* df, const void* dx);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template <>
void TetrahedronFEMForceField<CudaVec3fTypes>::reinit()
{
    this->core::componentmodel::behavior::ForceField<CudaVec3fTypes>::reinit();
    /// \TODO TetrahedronFEMForceField<CudaVec3fTypes>::reinit()
}

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    f.resize(x.size());
    TetrahedronFEMForceFieldCuda3f_addForce(data.nbVertex,
            data.nbElementPerVertex,
            data.elems.deviceRead(),
            data.state.deviceWrite(),
            data.velems.deviceRead(),
            (      Deriv*)f.deviceWrite() + data.vertex0,
            (const Coord*)x.deviceRead()  + data.vertex0,
            (const Deriv*)v.deviceRead()  + data.vertex0);
}

template <>
void TetrahedronFEMForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx)
{
    df.resize(dx.size());
    TetrahedronFEMForceFieldCuda3f_addDForce(data.nbVertex,
            data.nbElementPerVertex,
            data.elems.deviceRead(),
            data.state.deviceRead(),
            data.velems.deviceRead(),
            (      Deriv*)df.deviceWrite() + data.vertex0,
            (const Deriv*)dx.deviceRead()  + data.vertex0);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
