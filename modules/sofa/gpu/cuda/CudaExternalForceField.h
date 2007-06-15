#ifndef SOFA_GPU_CUDA_CUDAEXTERNALFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAEXTERNALFORCEFIELD_H

#include <sofa/component/interactionforcefield/ExternalForceField.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;
/** Apply given forces to given particles
*/
template <>
void ExternalForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);



} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
