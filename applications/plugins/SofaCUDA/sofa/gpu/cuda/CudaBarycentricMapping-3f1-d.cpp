/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "CudaBarycentricMapping.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

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

// Spread the instanciations over multiple files for more efficient and lightweight compilation

// instanciations involving both CudaVec3f1Types and Vec3dTypes



#ifndef SOFA_FLOAT
template class BarycentricMapping< Vec3dTypes, CudaVec3f1Types>;
template class BarycentricMapping< CudaVec3f1Types, Vec3dTypes>;
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

SOFA_DECL_CLASS(CudaBarycentricMapping_3f1_d)

int BarycentricMappingCudaClass_3f1_d = core::RegisterObject("Supports GPU-side computations using CUDA")
#ifndef SOFA_FLOAT
        .add< BarycentricMapping< Vec3dTypes, CudaVec3f1Types> >()
        .add< BarycentricMapping< CudaVec3f1Types, Vec3dTypes> >()
#endif
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
