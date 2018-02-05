/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include "OpenCLTypes.h"
#include <sofa/core/ObjectFactory.h>
#include <SofaEngine/BoxROI.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

template class boxroi::BoxROI<gpu::opencl::OpenCLVec3fTypes>;
template class boxroi::BoxROI<gpu::opencl::OpenCLVec3f1Types>;
template class boxroi::BoxROI<gpu::opencl::OpenCLVec3dTypes>;
template class boxroi::BoxROI<gpu::opencl::OpenCLVec3d1Types>;

} // namespace engine

} // namespace component

namespace gpu
{

namespace opencl
{

SOFA_DECL_CLASS(OpenCLBoxROI)

int BoxROIOpenCLClass = core::RegisterObject("Supports GPU-side computations using OPENCL")
        .add< component::engine::boxroi::BoxROI<OpenCLVec3fTypes> >()
        .add< component::engine::boxroi::BoxROI<OpenCLVec3f1Types> >()
        .add< component::engine::boxroi::BoxROI<OpenCLVec3dTypes> >()
        .add< component::engine::boxroi::BoxROI<OpenCLVec3d1Types> >()
        ;

} // namespace opencl

} // namespace gpu

} // namespace sofa
