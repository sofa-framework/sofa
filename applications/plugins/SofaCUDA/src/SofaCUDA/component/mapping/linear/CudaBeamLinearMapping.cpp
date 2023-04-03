/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/mapping/linear/BeamLinearMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa::gpu::cuda
{

using namespace sofa::component::mapping::linear;
using namespace defaulttype;
using namespace core;
using namespace core::behavior;


// Register in the Factory
int BeamLinearMappingCudaClass = core::RegisterObject("Set the positions and velocities of points attached to a beam using linear interpolation between DOFs")

        .add< BeamLinearMapping<Rigid3Types, CudaVec3Types> >()

#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< BeamLinearMapping<Rigid3Types, CudaVec3dTypes> >()
#endif
        ;

} // namespace sofa::gpu::cuda

namespace sofa::component::mapping::linear
{

using namespace defaulttype;
using namespace core;
using namespace core::behavior;

template class SOFA_GPU_CUDA_API BeamLinearMapping< Rigid3Types, sofa::gpu::cuda::CudaVec3Types>;


#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API BeamLinearMapping< Rigid3Types, sofa::gpu::cuda::CudaVec3dTypes>;
#endif


} // namespace sofa::component::mapping::linear
