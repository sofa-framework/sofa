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
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/topology/container/dynamic/PointSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/PointSetGeometryAlgorithms.inl>

#include <sofa/component/topology/container/dynamic/EdgeSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/EdgeSetGeometryAlgorithms.inl>

#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.inl>

#include <sofa/component/topology/container/dynamic/QuadSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/QuadSetGeometryAlgorithms.inl>

#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.inl>

#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetGeometryAlgorithms.inl>


namespace sofa::component::topology::container::dynamic
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::gpu::cuda;
using namespace sofa::core::behavior;

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

int CudaPointSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< PointSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< PointSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< PointSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API PointSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API PointSetGeometryAlgorithms<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API PointSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////      Edge      ////////
////////////////////////////////

int CudaEdgeSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< EdgeSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< EdgeSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API EdgeSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API EdgeSetGeometryAlgorithms<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API EdgeSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////    Triangle    ////////
////////////////////////////////

int CudaTriangleSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< TriangleSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< TriangleSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API TriangleSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API TriangleSetGeometryAlgorithms<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API TriangleSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////   Quad   ////////
////////////////////////////////

int CudaQuadSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< QuadSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< QuadSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< QuadSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API QuadSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API QuadSetGeometryAlgorithms<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API QuadSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////   Tetrahedron  ////////
////////////////////////////////

int CudaTetrahedronSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< TetrahedronSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< TetrahedronSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API TetrahedronSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API TetrahedronSetGeometryAlgorithms<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API TetrahedronSetGeometryAlgorithms<CudaVec3f1Types>;


////////////////////////////////
////////   Hexahedron   ////////
////////////////////////////////

int CudaHexahedronSetGeometryAlgorithmsClass = core::RegisterObject("")
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< HexahedronSetGeometryAlgorithms<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        .add< HexahedronSetGeometryAlgorithms<CudaVec3fTypes> >()
        .add< HexahedronSetGeometryAlgorithms<CudaVec3f1Types> >()
        ;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API HexahedronSetGeometryAlgorithms<CudaVec3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API HexahedronSetGeometryAlgorithms<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API HexahedronSetGeometryAlgorithms<CudaVec3f1Types>;

} // namespace sofa::component::topology::container::dynamic
