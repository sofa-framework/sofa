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
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaBaseTopology/PointSetTopologyAlgorithms.h>
#include <SofaBaseTopology/PointSetTopologyAlgorithms.inl>
#include <SofaBaseTopology/PointSetGeometryAlgorithms.h>
#include <SofaBaseTopology/PointSetGeometryAlgorithms.inl>

#include <SofaBaseTopology/EdgeSetTopologyAlgorithms.h>
#include <SofaBaseTopology/EdgeSetTopologyAlgorithms.inl>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.h>
#include <SofaBaseTopology/EdgeSetGeometryAlgorithms.inl>

#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.inl>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.inl>

#include <SofaBaseTopology/QuadSetTopologyAlgorithms.h>
#include <SofaBaseTopology/QuadSetTopologyAlgorithms.inl>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.h>
#include <SofaBaseTopology/QuadSetGeometryAlgorithms.inl>

#include <SofaBaseTopology/TetrahedronSetTopologyAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetTopologyAlgorithms.inl>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.inl>

#include <SofaBaseTopology/HexahedronSetTopologyAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetTopologyAlgorithms.inl>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/HexahedronSetGeometryAlgorithms.inl>


namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::gpu::cuda;
using namespace sofa::core::behavior;

SOFA_DECL_CLASS(CudaSetTopology)


/*
/// Cross product for 3-elements vectors.
template< class Real>
Real areaProduct(const Vec3r1<Real>& a, const Vec3r1<Real>& b)
{
    return Vec<3,Real>(a.y()*b.z() - a.z()*b.y(),
		       a.z()*b.x() - a.x()*b.z(),
		       a.x()*b.y() - a.y()*b.x()).norm();
}


/// Volume (triple product) for 3-elements vectors.
template<typename real>
inline real tripleProduct(const Vec3r1<real>& a, const Vec3r1<real>& b,const Vec3r1<real> &c)
{
    return dot(a,cross(b,c));
}
*/

////////////////////////////////
////////     Point      ////////
////////////////////////////////

int CudaPointSetTopologyAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< PointSetTopologyAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< PointSetTopologyAlgorithms<CudaVec3fTypes> >()
        .add< PointSetTopologyAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class PointSetTopologyAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class PointSetTopologyAlgorithms<CudaVec3fTypes>;
template class PointSetTopologyAlgorithms<CudaVec3f1Types>;


int CudaPointSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< PointSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< PointSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< PointSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class PointSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class PointSetGeometryAlgorithms<CudaVec3fTypes>;
template class PointSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////      Edge      ////////
////////////////////////////////

int CudaEdgeSetTopologyAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetTopologyAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetTopologyAlgorithms<CudaVec3fTypes> >()
        .add< EdgeSetTopologyAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class EdgeSetTopologyAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class EdgeSetTopologyAlgorithms<CudaVec3fTypes>;
template class EdgeSetTopologyAlgorithms<CudaVec3f1Types>;


int CudaEdgeSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< EdgeSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class EdgeSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class EdgeSetGeometryAlgorithms<CudaVec3fTypes>;
template class EdgeSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////    Triangle    ////////
////////////////////////////////

int CudaTriangleSetTopologyAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetTopologyAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetTopologyAlgorithms<CudaVec3fTypes> >()
        .add< TriangleSetTopologyAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class TriangleSetTopologyAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class TriangleSetTopologyAlgorithms<CudaVec3fTypes>;
template class TriangleSetTopologyAlgorithms<CudaVec3f1Types>;


int CudaTriangleSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< TriangleSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class TriangleSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class TriangleSetGeometryAlgorithms<CudaVec3fTypes>;
template class TriangleSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////   Quad   ////////
////////////////////////////////

int CudaQuadSetTopologyAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< QuadSetTopologyAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< QuadSetTopologyAlgorithms<CudaVec3fTypes> >()
        .add< QuadSetTopologyAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class QuadSetTopologyAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class QuadSetTopologyAlgorithms<CudaVec3fTypes>;
template class QuadSetTopologyAlgorithms<CudaVec3f1Types>;


int CudaQuadSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< QuadSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< QuadSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< QuadSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class QuadSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class QuadSetGeometryAlgorithms<CudaVec3fTypes>;
template class QuadSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////   Tetrahedron  ////////
////////////////////////////////

int CudaTetrahedronSetTopologyAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetTopologyAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetTopologyAlgorithms<CudaVec3fTypes> >()
        .add< TetrahedronSetTopologyAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class TetrahedronSetTopologyAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class TetrahedronSetTopologyAlgorithms<CudaVec3fTypes>;
template class TetrahedronSetTopologyAlgorithms<CudaVec3f1Types>;


int CudaTetrahedronSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< TetrahedronSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class TetrahedronSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class TetrahedronSetGeometryAlgorithms<CudaVec3fTypes>;
template class TetrahedronSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////   Hexahedron   ////////
////////////////////////////////

int CudaHexahedronSetTopologyAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< HexahedronSetTopologyAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< HexahedronSetTopologyAlgorithms<CudaVec3fTypes> >()
        .add< HexahedronSetTopologyAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class HexahedronSetTopologyAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class HexahedronSetTopologyAlgorithms<CudaVec3fTypes>;
template class HexahedronSetTopologyAlgorithms<CudaVec3f1Types>;


int CudaHexahedronSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< HexahedronSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< HexahedronSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< HexahedronSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class HexahedronSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class HexahedronSetGeometryAlgorithms<CudaVec3fTypes>;
template class HexahedronSetGeometryAlgorithms<CudaVec3f1Types>;


} // namespace topology

} // namespace component

} // namespace sofa
