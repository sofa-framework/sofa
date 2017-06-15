/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMiscMapping/BeamLinearMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>
#include "CudaTypes.h"

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaBeamLinearMapping)

using namespace sofa::component::mapping;
using namespace defaulttype;
using namespace core;
using namespace core::behavior;


// Register in the Factory
int BeamLinearMappingCudaClass = core::RegisterObject("Set the positions and velocities of points attached to a beam using linear interpolation between DOFs")

        .add< BeamLinearMapping<Rigid3fTypes, CudaVec3fTypes> >()
#ifndef SOFA_FLOAT
        .add< BeamLinearMapping<Rigid3dTypes, CudaVec3fTypes> >()
#endif
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< BeamLinearMapping<Rigid3fTypes, CudaVec3dTypes> >()
        .add< BeamLinearMapping<Rigid3dTypes, CudaVec3dTypes> >()
#endif
        ;

}

}

namespace component
{

namespace mapping
{

using namespace defaulttype;
using namespace core;
using namespace core::behavior;

template class BeamLinearMapping< Rigid3fTypes, sofa::gpu::cuda::CudaVec3fTypes>;
#ifndef SOFA_FLOAT
template class BeamLinearMapping< Rigid3dTypes, sofa::gpu::cuda::CudaVec3fTypes>;
#endif

#ifdef SOFA_GPU_CUDA_DOUBLE
template class BeamLinearMapping< Rigid3fTypes, sofa::gpu::cuda::CudaVec3dTypes>;
template class BeamLinearMapping< Rigid3dTypes, sofa::gpu::cuda::CudaVec3dTypes>;
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

