/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/TetrahedronSetTopology.inl>
#include <sofa/component/topology/EdgeSetTopology.inl>
#include <sofa/component/topology/TriangleSetTopology.inl>
#include <sofa/component/topology/PointSetTopology.inl>
//#include <sofa/component/topology/Tetra2TriangleTopologicalMapping.inl>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::gpu::cuda;
using namespace sofa::core::componentmodel::behavior;

SOFA_DECL_CLASS(CudaSetTopology)

//PointSetTopology
int PointSetTopologyCudaClass = core::RegisterObject("Topology consisting of a set of points")
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< PointSetTopology<CudaVec3dTypes> >()
//    .add< PointSetTopology<CudaVec2dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        .add< PointSetTopology<CudaVec3fTypes> >()
//    .add< PointSetTopology<CudaVec2fTypes> >()
        ;

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
template class PointSetTopologyModifier<CudaVec3dTypes>;
//template class PointSetTopologyModifier<CudaVec2dTypes>;
template class PointSetTopology<CudaVec3dTypes>;
//template class PointSetTopology<CudaVec2dTypes>;
template class PointSetGeometryAlgorithms<CudaVec3dTypes>;
//template class PointSetGeometryAlgorithms<CudaVec2dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

template class PointSetTopologyModifier<CudaVec3fTypes>;
//template class PointSetTopologyModifier<CudaVec2fTypes>;
template class PointSetTopology<CudaVec3fTypes>;
//template class PointSetTopology<CudaVec2fTypes>;
template class PointSetGeometryAlgorithms<CudaVec3fTypes>;
//template class PointSetGeometryAlgorithms<CudaVec2fTypes>;

//EdgeSetTopology
int EdgeSetTopologyCudaClass = core::RegisterObject("Dynamic topology handling point sets")
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetTopology<CudaVec3dTypes> >()
//.add< EdgeSetTopology<CudaVec2dTypes> >()
//.add< EdgeSetTopology<CudaVec1dTypes> >()
        .add< EdgeSetTopology<CudaRigid3dTypes> >()
//.add< EdgeSetTopology<CudaRigid2dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        .add< EdgeSetTopology<CudaVec3fTypes> >()
//.add< EdgeSetTopology<CudaVec2fTypes> >()
//.add< EdgeSetTopology<CudaVec1fTypes> >()
        .add< EdgeSetTopology<CudaRigid3fTypes> >()
//.add< EdgeSetTopology<CudaRigid2fTypes> >()
        ;


#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
template class EdgeSetTopology<CudaVec3dTypes>;
//template class EdgeSetTopology<CudaVec2dTypes>;
//template class EdgeSetTopology<CudaVec1dTypes>;
template class EdgeSetTopology<CudaRigid3dTypes>;
//template class EdgeSetTopology<CudaRigid2dTypes>;

template class EdgeSetTopologyAlgorithms<CudaVec3dTypes>;
//template class EdgeSetTopologyAlgorithms<CudaVec2dTypes>;
//template class EdgeSetTopologyAlgorithms<CudaVec1dTypes>;
template class EdgeSetTopologyAlgorithms<CudaRigid3dTypes>;
//template class EdgeSetTopologyAlgorithms<CudaRigid2dTypes>;

template class EdgeSetGeometryAlgorithms<CudaVec3dTypes>;
//template class EdgeSetGeometryAlgorithms<CudaVec2dTypes>;
//template class EdgeSetGeometryAlgorithms<CudaVec1dTypes>;
template class EdgeSetGeometryAlgorithms<CudaRigid3dTypes>;
//template class EdgeSetGeometryAlgorithms<CudaRigid2dTypes>;

template class EdgeSetTopologyModifier<CudaVec3dTypes>;
//template class EdgeSetTopologyModifier<CudaVec2dTypes>;
//template class EdgeSetTopologyModifier<CudaVec1dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

template class EdgeSetTopology<CudaVec3fTypes>;
//template class EdgeSetTopology<CudaVec2fTypes>;
//template class EdgeSetTopology<CudaVec1fTypes>;
template class EdgeSetTopology<CudaRigid3fTypes>;
//template class EdgeSetTopology<CudaRigid2fTypes>;

template class EdgeSetTopologyAlgorithms<CudaVec3fTypes>;
//template class EdgeSetTopologyAlgorithms<CudaVec2fTypes>;
//template class EdgeSetTopologyAlgorithms<CudaVec1fTypes>;
template class EdgeSetTopologyAlgorithms<CudaRigid3fTypes>;
//template class EdgeSetTopologyAlgorithms<CudaRigid2fTypes>;


template class EdgeSetGeometryAlgorithms<CudaVec3fTypes>;
//template class EdgeSetGeometryAlgorithms<CudaVec2fTypes>;
//template class EdgeSetGeometryAlgorithms<CudaVec1fTypes>;
template class EdgeSetGeometryAlgorithms<CudaRigid3fTypes>;
//template class EdgeSetGeometryAlgorithms<CudaRigid2fTypes>;

template class EdgeSetTopologyModifier<CudaVec3fTypes>;
//template class EdgeSetTopologyModifier<CudaVec2fTypes>;
//template class EdgeSetTopologyModifier<CudaVec1fTypes>;

//TriangleSetTopology

int TriangleSetTopologyCudaClass = core::RegisterObject("Triangle set topology")
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetTopology<CudaVec3dTypes> >()
//.add< TriangleSetTopology<CudaVec2dTypes> >()
//.add< TriangleSetTopology<CudaVec1dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        .add< TriangleSetTopology<CudaVec3fTypes> >()
//.add< TriangleSetTopology<CudaVec2fTypes> >()
//.add< TriangleSetTopology<CudaVec1fTypes> >()
        ;


#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
template class TriangleSetTopology<CudaVec3dTypes>;
//template class TriangleSetTopology<CudaVec2dTypes>;
//template class TriangleSetTopology<CudaVec1dTypes>;

template class TriangleSetTopologyAlgorithms<CudaVec3dTypes>;
//template class TriangleSetTopologyAlgorithms<CudaVec2dTypes>;
//template class TriangleSetTopologyAlgorithms<CudaVec1dTypes>;

template class TriangleSetGeometryAlgorithms<CudaVec3dTypes>;
//template class TriangleSetGeometryAlgorithms<CudaVec2dTypes>;
//template class TriangleSetGeometryAlgorithms<CudaVec1dTypes>;


template class TriangleSetTopologyModifier<CudaVec3dTypes>;
//template class TriangleSetTopologyModifier<CudaVec2dTypes>;
//template class TriangleSetTopologyModifier<CudaVec1dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

template class TriangleSetTopology<CudaVec3fTypes>;
//template class TriangleSetTopology<CudaVec2fTypes>;
//template class TriangleSetTopology<CudaVec1fTypes>;

template class TriangleSetTopologyAlgorithms<CudaVec3fTypes>;
//template class TriangleSetTopologyAlgorithms<CudaVec2fTypes>;
//template class TriangleSetTopologyAlgorithms<CudaVec1fTypes>;

template class TriangleSetGeometryAlgorithms<CudaVec3fTypes>;
//template class TriangleSetGeometryAlgorithms<CudaVec2fTypes>;
//template class TriangleSetGeometryAlgorithms<CudaVec1fTypes>;

template class TriangleSetTopologyModifier<CudaVec3fTypes>;
//template class TriangleSetTopologyModifier<CudaVec2fTypes>;
//template class TriangleSetTopologyModifier<CudaVec1fTypes>;

//TetrahedronSetTopology

int TetrahedronSetTopologyCudaClass = core::RegisterObject("Tetrahedron set topology")
#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetTopology<CudaVec3dTypes> >()
//.add< TetrahedronSetTopology<CudaVec2dTypes> >()
//.add< TetrahedronSetTopology<CudaVec1dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV
        .add< TetrahedronSetTopology<CudaVec3fTypes> >()
//.add< TetrahedronSetTopology<CudaVec2fTypes> >()
//.add< TetrahedronSetTopology<CudaVec1fTypes> >()
        ;


#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE
template class TetrahedronSetTopology<CudaVec3dTypes>;
//template class TetrahedronSetTopology<CudaVec2dTypes>;
//template class TetrahedronSetTopology<CudaVec1dTypes>;

template class TetrahedronSetTopologyAlgorithms<CudaVec3dTypes>;
//template class TetrahedronSetTopologyAlgorithms<CudaVec2dTypes>;
//template class TetrahedronSetTopologyAlgorithms<CudaVec1dTypes>;

template class TetrahedronSetGeometryAlgorithms<CudaVec3dTypes>;
//template class TetrahedronSetGeometryAlgorithms<CudaVec2dTypes>;
//template class TetrahedronSetGeometryAlgorithms<CudaVec1dTypes>;

template class TetrahedronSetTopologyModifier<CudaVec3dTypes>;
//template class TetrahedronSetTopologyModifier<CudaVec2dTypes>;
//template class TetrahedronSetTopologyModifier<CudaVec1dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

template class TetrahedronSetTopology<CudaVec3fTypes>;
//template class TetrahedronSetTopology<CudaVec2fTypes>;
//template class TetrahedronSetTopology<CudaVec1fTypes>;

template class TetrahedronSetTopologyAlgorithms<CudaVec3fTypes>;
//template class TetrahedronSetTopologyAlgorithms<CudaVec2fTypes>;
//template class TetrahedronSetTopologyAlgorithms<CudaVec1fTypes>;

template class TetrahedronSetGeometryAlgorithms<CudaVec3fTypes>;
//template class TetrahedronSetGeometryAlgorithms<CudaVec2fTypes>;
//template class TetrahedronSetGeometryAlgorithms<CudaVec1fTypes>;

template class TetrahedronSetTopologyModifier<CudaVec3fTypes>;
//template class TetrahedronSetTopologyModifier<CudaVec2fTypes>;
//template class TetrahedronSetTopologyModifier<CudaVec1fTypes>;


//int Tetra2TriangleTopologicalMappingCudaClass = core::RegisterObject("Special case of mapping where TetrahedronSetTopology is converted to TriangleSetTopology")
//.add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<CudaVec3fTypes>, TriangleSetTopology<Vec3dTypes> > >()
//.add< Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<CudaVec3fTypes>, TriangleSetTopology<Vec3fTypes> > >()
//;
//
//template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<CudaVec3fTypes>, TriangleSetTopology<Vec3dTypes> >;
//template class Tetra2TriangleTopologicalMapping< TetrahedronSetTopology<CudaVec3fTypes>, TriangleSetTopology<Vec3fTypes> >;


} // namespace topology

} // namespace component

} // namespace sofa
