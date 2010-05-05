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
#include "CudaTypes.h"
#include "CudaBarycentricMapping.inl"
//#include <sofa/component/mapping/BarycentricMapping.inl>
#include <sofa/core/behavior/MappedModel.h>
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
#ifndef SOFA_FLOAT
#endif
#ifndef SOFA_DOUBLE
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3fTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3fTypes> > >;
// template class BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3dTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3f1Types> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3f1Types> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3f1Types> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3f1Types> > >;
template class BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3fTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3f1Types> > >;
template class BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3fTypes> > >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3dTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<Vec3dTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3f1Types> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3dTypes> > >;
// template class BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3dTypes> > >;
template class BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3dTypes> > >;
#endif
#endif
}
}

namespace gpu
{

namespace cuda
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::component::mapping;

SOFA_DECL_CLASS(CudaBarycentricMapping)

int BarycentricMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
#ifndef SOFA_FLOAT
#endif
#ifndef SOFA_DOUBLE
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
// .add< BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3f1Types> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3f1Types> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<CudaVec3f1Types> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3fTypes> > > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3f1Types> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3f1Types>, MechanicalState<Vec3dTypes> > > >()
// .add< BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<ExtVec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<CudaVec3f1Types>, MappedModel<Vec3dTypes> > > >()
#endif
#endif


#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3fTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<CudaVec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<CudaVec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<CudaVec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<CudaVec3dTypes> > > >()
//.add< BarycentricMapping< Mapping< State<CudaVec3d1Types>, MappedModel<ExtVec3fTypes> > > >()
//.add< BarycentricMapping< Mapping< State<CudaVec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
