/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "CudaTypes.h"

#include <sofa/core/ObjectFactory.h>

#include <sofa/defaulttype/defaulttype.h>
#include <sofa/core/MultiMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

using namespace sofa::defaulttype;
using namespace sofa::core;

template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3fTypes, Vec1dTypes>;
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3fTypes, Vec3dTypes>;
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3fTypes, Vec1fTypes>;
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3fTypes, Vec3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, Vec1dTypes>;
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, Vec3dTypes>;
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, Vec1fTypes>;
template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, Vec3fTypes>;
#endif

} // namespace cuda

} // namespace gpu

} // namespace sofa
