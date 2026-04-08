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
#include <SofaCUDA/component/config.h>

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/solidmechanics/fem/elastic/CudaElementCorotationalFEMForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::fem::elastic
{

using namespace sofa::gpu::cuda;

template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Edge>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Triangle>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Quad>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Tetrahedron>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Hexahedron>;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Edge>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Triangle>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Quad>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Tetrahedron>;
template class SOFACUDA_COMPONENT_API CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Hexahedron>;
#endif

} // namespace sofa::component::solidmechanics::fem::elastic

namespace sofa::gpu::cuda
{

void registerElementCorotationalFEMForceField(sofa::core::ObjectFactory* factory)
{
    using namespace sofa::component::solidmechanics::fem::elastic;

    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA for EdgeCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Edge> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA for TriangleCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Triangle> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA for QuadCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Quad> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA for TetrahedronCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Tetrahedron> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA for HexahedronCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3fTypes, sofa::geometry::Hexahedron> >()
    );

#ifdef SOFA_GPU_CUDA_DOUBLE
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA (double) for EdgeCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Edge> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA (double) for TriangleCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Triangle> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA (double) for QuadCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Quad> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA (double) for TetrahedronCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Tetrahedron> >()
    );
    factory->registerObjects(sofa::core::ObjectRegistrationData(
        "Supports GPU-side computations using CUDA (double) for HexahedronCorotationalFEMForceField")
        .add< CudaElementCorotationalFEMForceField<CudaVec3dTypes, sofa::geometry::Hexahedron> >()
    );
#endif
}

} // namespace sofa::gpu::cuda
