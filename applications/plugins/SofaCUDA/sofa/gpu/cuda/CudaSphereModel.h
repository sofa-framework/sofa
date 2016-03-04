/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDASPHEREMODEL_H
#define SOFA_GPU_CUDA_CUDASPHEREMODEL_H

#include "CudaTypes.h"

#include <SofaBaseCollision/SphereModel.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

typedef sofa::component::collision::TSphereModel<gpu::cuda::CudaVec3Types> CudaSphereModel;
typedef sofa::component::collision::TSphere<gpu::cuda::CudaVec3Types> CudaSphere;


} // namespace cuda

} // namespace gpu


namespace component
{
namespace collision
{


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_GPU_CUDA)
extern template class SOFA_GPU_CUDA_API sofa::component::collision::TSphereModel<sofa::gpu::cuda::CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API sofa::component::collision::TSphereModel<sofa::gpu::cuda::CudaVec3f1Types>;
#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API sofa::component::collision::TSphereModel<sofa::gpu::cuda::CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API sofa::component::collision::TSphereModel<sofa::gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif

} // namespace collision

} // namespace component


} // namespace sofa

#endif
