/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/mapping/BeamLinearMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>
#include <sofa/core/behavior/MechanicalMapping.inl>
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

        .add< BeamLinearMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BeamLinearMapping< Mapping< State<Rigid3fTypes>, MappedModel<CudaVec3fTypes> > > >()
        .add< BeamLinearMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BeamLinearMapping< Mapping< State<Rigid3dTypes>, MappedModel<CudaVec3fTypes> > > >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< BeamLinearMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< BeamLinearMapping< Mapping< State<Rigid3fTypes>, MappedModel<CudaVec3dTypes> > > >()
        .add< BeamLinearMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< BeamLinearMapping< Mapping< State<Rigid3dTypes>, MappedModel<CudaVec3dTypes> > > >()
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

template class BeamLinearMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<sofa::gpu::cuda::CudaVec3fTypes> > >;
template class BeamLinearMapping< Mapping< State<Rigid3fTypes>, MappedModel<sofa::gpu::cuda::CudaVec3fTypes> > >;
template class BeamLinearMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<sofa::gpu::cuda::CudaVec3fTypes> > >;
template class BeamLinearMapping< Mapping< State<Rigid3dTypes>, MappedModel<sofa::gpu::cuda::CudaVec3fTypes> > >;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class BeamLinearMapping< MechanicalMapping<MechanicalState<Rigid3fTypes>, MechanicalState<sofa::gpu::cuda::CudaVec3dTypes> > >;
template class BeamLinearMapping< Mapping< State<Rigid3fTypes>, MappedModel<sofa::gpu::cuda::CudaVec3dTypes> > >;
template class BeamLinearMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<sofa::gpu::cuda::CudaVec3dTypes> > >;
template class BeamLinearMapping< Mapping< State<Rigid3dTypes>, MappedModel<sofa::gpu::cuda::CudaVec3dTypes> > >;
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

