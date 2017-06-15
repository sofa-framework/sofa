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
#include "CudaTypes.h"
#include "CudaSubsetMapping.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::cuda;

template class SubsetMapping< CudaVec3fTypes, CudaVec3fTypes >;
// template class SubsetMapping< CudaVec3fTypes, ExtVec3dTypes >;
template class SubsetMapping< CudaVec3fTypes, ExtVec3fTypes >;
template class SubsetMapping< CudaVec3f1Types, CudaVec3f1Types >;
template class SubsetMapping< CudaVec3f1Types, CudaVec3fTypes >;
template class SubsetMapping< CudaVec3fTypes, CudaVec3f1Types >;
template class SubsetMapping< CudaVec3f1Types, ExtVec3fTypes >;
// template class SubsetMapping< CudaVec3f1Types, ExtVec3dTypes >;
#ifndef SOFA_FLOAT
#endif
#ifndef SOFA_DOUBLE
template class SubsetMapping< CudaVec3fTypes, Vec3fTypes >;
template class SubsetMapping< CudaVec3f1Types, Vec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SubsetMapping< CudaVec3fTypes, Vec3dTypes >;
template class SubsetMapping< CudaVec3f1Types, Vec3dTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

namespace gpu
{

namespace cuda
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(CudaSubsetMapping)

int SubsetMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< SubsetMapping< CudaVec3fTypes, CudaVec3fTypes > >()
        .add< SubsetMapping< CudaVec3f1Types, CudaVec3f1Types > >()
        .add< SubsetMapping< CudaVec3f1Types, CudaVec3fTypes > >()
        .add< SubsetMapping< CudaVec3fTypes, CudaVec3f1Types > >()
        .add< SubsetMapping< CudaVec3f1Types, ExtVec3fTypes > >()
        .add< SubsetMapping< CudaVec3fTypes, ExtVec3fTypes > >()
// .add< SubsetMapping< CudaVec3fTypes, ExtVec3dTypes > >()
// .add< SubsetMapping< CudaVec3f1Types, ExtVec3dTypes > >()
#ifndef SOFA_FLOAT
#endif
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< CudaVec3fTypes, Vec3fTypes > >()
        .add< SubsetMapping< CudaVec3f1Types, Vec3fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< SubsetMapping< CudaVec3fTypes, Vec3dTypes > >()
        .add< SubsetMapping< CudaVec3f1Types, Vec3dTypes > >()
#endif
#endif
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
