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
#include <SofaCUDA/component/mapping/linear/CudaBarycentricMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::cuda;


////////////////////////////////////////////////////////////
//////////          RegularGridTopology           //////////
////////////////////////////////////////////////////////////


template<>
void BarycentricMapperRegularGridTopology<CudaVec3f1Types,CudaVec3f1Types>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    const unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f1_apply(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    const unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f1_applyJ(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    calcMapT();
    if (map.size() == 0) return;
    const unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
    const unsigned int insize = out.size();

    RegularGridMapperCuda3f1_applyJT(insize, maxNOut, gridsize, mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJT( In::MatrixDeriv& /*out*/, const Out::MatrixDeriv& /*in*/ )
{
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3f1Types,CudaVec3f1Types>::draw (const core::visual::VisualParams* ,const Out::VecCoord& /*out*/, const In::VecCoord& /*in*/)
{
}


template<>
void BarycentricMapperRegularGridTopology<CudaVec3f1Types,CudaVec3f1Types>::resize( core::State<Out>* toModel )
{
    SOFA_UNUSED(toModel);
}


////////////////////////////////////////////////////////////////////////////
//////////          BarycentricMapperSparseGridTopology           //////////
////////////////////////////////////////////////////////////////////////////


template<>
void BarycentricMapperSparseGridTopology<CudaVec3f1Types,CudaVec3f1Types>::draw (const core::visual::VisualParams* ,const Out::VecCoord& /*out*/, const In::VecCoord& /*in*/)
{
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3f1Types,CudaVec3f1Types>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(map.size());
    buildHexa();
    SparseGridMapperCuda3f1_apply(map.size(), CudaHexa.deviceRead(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(map.size());
    buildHexa();
    SparseGridMapperCuda3f1_applyJ(map.size(), CudaHexa.deviceRead(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    buildTranslate(out.size());
    SparseGridMapperCuda3f1_applyJT(out.size(), CudaTnb.deviceRead(),CudaTst.deviceRead(),CudaTid.deviceRead(),CudaTVal.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in)
{
    const helper::ReadAccessor<gpu::cuda::CudaVector<CubeData> > map = this->map;

    for (Out::MatrixDeriv::RowConstIterator rowIt = in.begin(), rowItEnd = in.end(); rowIt != rowItEnd; ++rowIt)
    {
        Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                const unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

                const auto cube = this->m_fromTopology->getHexahedron ( map[indexIn].in_index );

                const OutReal fx = ( OutReal ) map[indexIn].baryCoords[0];
                const OutReal fy = ( OutReal ) map[indexIn].baryCoords[1];
                const OutReal fz = ( OutReal ) map[indexIn].baryCoords[2];
                const OutReal oneMinusFx = 1-fx;
                const OutReal oneMinusFy = 1-fy;
                const OutReal oneMinusFz = 1-fz;

                OutReal f = ( oneMinusFx * oneMinusFy * oneMinusFz );
                o.addCol ( cube[0],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * oneMinusFz );
                o.addCol ( cube[1],  ( data * f ) );

                f = ( oneMinusFx * ( fy ) * oneMinusFz );
                o.addCol ( cube[3],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * oneMinusFz );
                o.addCol ( cube[2],  ( data * f ) );

                f = ( oneMinusFx * oneMinusFy * ( fz ) );
                o.addCol ( cube[4],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * ( fz ) );
                o.addCol ( cube[5],  ( data * f ) );

                f = ( oneMinusFx * ( fy ) * ( fz ) );
                o.addCol ( cube[7],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * ( fz ) );
                o.addCol ( cube[6],  ( data * f ) );
            }
        }
    }
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3f1Types,CudaVec3f1Types>::resize( core::State<Out>* toModel )
{
    SOFA_UNUSED(toModel);
}



////////////////////////////////////////////////////////////
//////////            BaseMeshTopology            //////////
////////////////////////////////////////////////////////////

template<>
void BarycentricMapperMeshTopology<CudaVec3f1Types,CudaVec3f1Types>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(size);
    MeshMapperCuda3f1_apply(size, maxNIn, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperMeshTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(size);
    MeshMapperCuda3f1_apply(size, maxNIn, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperMeshTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    if (size == 0) return;
    calcMapT();
    MeshMapperCuda3f1_applyPEq(insize, maxNOut, mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperMeshTopology<CudaVec3f1Types,CudaVec3f1Types>::applyJT( In::MatrixDeriv& /*out*/, const Out::MatrixDeriv& /*in*/ )
{
}

template<>
void BarycentricMapperMeshTopology<CudaVec3f1Types,CudaVec3f1Types>::draw (const core::visual::VisualParams* ,const Out::VecCoord& /*out*/, const In::VecCoord& /*in*/)
{
}

template<>
void BarycentricMapperMeshTopology<CudaVec3f1Types,CudaVec3f1Types>::resize( core::State<Out>* toModel )
{
    SOFA_UNUSED(toModel);
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Spread the instanciations over multiple files for more efficient and lightweight compilation

// instanciations involving only CudaVec3f1Types with CudaVec3f1Types

template class SOFA_GPU_CUDA_API BarycentricMapping< CudaVec3f1Types, CudaVec3f1Types>;



} // namespace sofa::component::mapping::linear

namespace sofa::gpu::cuda
{
using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::component::mapping::linear;

int BarycentricMappingCudaClass_3f1 = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< BarycentricMapping< CudaVec3f1Types, CudaVec3f1Types> >()
        ;

} // namespace sofa::gpu::cuda
