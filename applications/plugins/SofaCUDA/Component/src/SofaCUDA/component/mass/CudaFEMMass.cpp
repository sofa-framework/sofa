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
#define SOFA_GPU_CUDA_CUDAFEMMASS_CPP

#include <SofaCUDA/component/config.h>

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/mass/CudaFEMMass.h>
#include <sofa/component/mass/FEMMass.inl>
#include <sofa/core/behavior/Mass.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/fem/FiniteElement[all].h>

namespace sofa::component::mass
{

using namespace sofa::gpu::cuda;

template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Edge>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Triangle>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Quad>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Tetrahedron>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Hexahedron>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Prism>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3fTypes, sofa::geometry::Pyramid>;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Edge>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Triangle>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Quad>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Tetrahedron>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Hexahedron>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Prism>;
template class SOFACUDA_COMPONENT_API FEMMass<CudaVec3dTypes, sofa::geometry::Pyramid>;
#endif

} // namespace sofa::component::mass

namespace sofa::gpu::cuda
{

void registerFEMMass(sofa::core::ObjectFactory* factory)
{
    using namespace sofa::component::mass;

    factory->registerObjects(sofa::core::ObjectRegistrationData("Supports GPU-side computations using CUDA for FEMMass")
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Edge> >()
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Triangle> >()
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Quad> >()
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Tetrahedron> >()
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Hexahedron> >()
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Prism> >()
        .add< FEMMass<CudaVec3fTypes, sofa::geometry::Pyramid> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Edge> >()
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Triangle> >()
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Quad> >()
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Tetrahedron> >()
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Hexahedron> >()
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Prism> >()
        .add< FEMMass<CudaVec3dTypes, sofa::geometry::Pyramid> >()
#endif
    );
}

} // namespace sofa::gpu::cuda
