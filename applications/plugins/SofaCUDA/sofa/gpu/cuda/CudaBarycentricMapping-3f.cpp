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


////////////////////////////////////////////////////////////
//////////          RegularGridTopology           //////////
////////////////////////////////////////////////////////////


template<>
void BarycentricMapperRegularGridTopology<CudaVec3fTypes,CudaVec3fTypes>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f_apply(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f_applyJ(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    if (map.size() == 0) return;
    calcMapT();
    unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
    unsigned int insize = out.size();

    RegularGridMapperCuda3f_applyJT(insize, maxNOut, gridsize, mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::MatrixDeriv& /*out*/, const Out::MatrixDeriv& /*in*/ )
{
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3fTypes,CudaVec3fTypes>::draw (const core::visual::VisualParams* ,const Out::VecCoord& /*out*/, const In::VecCoord& /*in*/)
{
}

template<>
void BarycentricMapperRegularGridTopology<CudaVec3fTypes,CudaVec3fTypes>::resize( core::State<Out>* /*toModel*/ )
{
//    toModel->resize(map.size());
}
////////////////////////////////////////////////////////////////////////////
//////////          BarycentricMapperSparseGridTopology           //////////
////////////////////////////////////////////////////////////////////////////

template<>
void BarycentricMapperSparseGridTopology<CudaVec3fTypes,CudaVec3fTypes>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(map.size());
    buildHexa();
    SparseGridMapperCuda3f_apply(map.size(), CudaHexa.deviceRead(),map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(map.size());
    buildHexa();
    SparseGridMapperCuda3f_applyJ(map.size(), CudaHexa.deviceRead(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    buildTranslate(out.size());
    SparseGridMapperCuda3f_applyJT(out.size(), CudaTnb.deviceRead(),CudaTst.deviceRead(),CudaTid.deviceRead(),CudaTVal.deviceRead(), out.deviceWrite(), in.deviceRead());
    /*
            for ( unsigned int i=0;i<map.size();i++ ) {
                    Out::Deriv v = in[i];
            const topology::SparseGridTopology::Hexa cube = this->topology->getHexahedron ( map[i].in_index );
                    const OutReal fx = ( OutReal ) map[i].baryCoords[0];
                    const OutReal fy = ( OutReal ) map[i].baryCoords[1];
                    const OutReal fz = ( OutReal ) map[i].baryCoords[2];
                    out[cube[0]] += v * ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) );
                    out[cube[1]] += v * ( (   fx ) * ( 1-fy ) * ( 1-fz ) );
                    out[cube[3]] += v * ( ( 1-fx ) * (   fy ) * ( 1-fz ) );
                    out[cube[2]] += v * ( (   fx ) * (   fy ) * ( 1-fz ) );
                    out[cube[4]] += v * ( ( 1-fx ) * ( 1-fy ) * (   fz ) );
                    out[cube[5]] += v * ( (   fx ) * ( 1-fy ) * (   fz ) );
                    out[cube[7]] += v * ( ( 1-fx ) * (   fy ) * (   fz ) );
                    out[cube[6]] += v * ( (   fx ) * (   fy ) * (   fz ) );
            }
    */
// 	for ( unsigned int o=0;o<out.size();o++ ) {
// 	    for (unsigned n=CudaTst[o];n<CudaTst[o]+CudaTnb[o];n++) {
// 	      out[o] += in[CudaTid[n]] * CudaTVal[n];
// 	    }
// 	}



}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in )
{
    helper::ReadAccessor<gpu::cuda::CudaVector<CubeData> > map = this->map;

    for (Out::MatrixDeriv::RowConstIterator rowIt = in.begin(), rowItEnd = in.end(); rowIt != rowItEnd; ++rowIt)
    {
        Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        if (colIt != colItEnd)
        {
            In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            for ( ; colIt != colItEnd; ++colIt)
            {
                unsigned indexIn = colIt.index();
                InDeriv data = (InDeriv) Out::getDPos(colIt.val());

#ifdef SOFA_NEW_HEXA
                const topology::SparseGridTopology::Hexa cube = this->fromTopology->getHexahedron ( map[indexIn].in_index );
#else
                const topology::SparseGridTopology::Cube cube = this->fromTopology->getCube ( map[indexIn].in_index );
#endif
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

#ifdef SOFA_NEW_HEXA
                f = ( oneMinusFx * ( fy ) * oneMinusFz );
                o.addCol ( cube[3],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * oneMinusFz );
                o.addCol ( cube[2],  ( data * f ) );

#else
                f = ( oneMinusFx * ( fy ) * oneMinusFz );
                o.addCol ( cube[2],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * oneMinusFz );
                o.addCol ( cube[3],  ( data * f ) );

#endif
                f = ( oneMinusFx * oneMinusFy * ( fz ) );
                o.addCol ( cube[4],  ( data * f ) );

                f = ( ( fx ) * oneMinusFy * ( fz ) );
                o.addCol ( cube[5],  ( data * f ) );

#ifdef SOFA_NEW_HEXA
                f = ( oneMinusFx * ( fy ) * ( fz ) );
                o.addCol ( cube[7],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * ( fz ) );
                o.addCol ( cube[6],  ( data * f ) );
#else
                f = ( oneMinusFx * ( fy ) * ( fz ) );
                o.addCol ( cube[6],  ( data * f ) );

                f = ( ( fx ) * ( fy ) * ( fz ) );
                o.addCol ( cube[7],  ( data * f ) );
#endif
            }
        }
    }
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3fTypes,CudaVec3fTypes>::draw (const core::visual::VisualParams* ,const Out::VecCoord& /*out*/, const In::VecCoord& /*in*/)
{
}

template<>
void BarycentricMapperSparseGridTopology<CudaVec3fTypes,CudaVec3fTypes>::resize( core::State<Out>* /*toModel*/ )
{
//    toModel->resize(map.size());
}

////////////////////////////////////////////////////////////
//////////            BaseMeshTopology            //////////
////////////////////////////////////////////////////////////

template<>
void BarycentricMapperMeshTopology<CudaVec3fTypes,CudaVec3fTypes>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(size);
    MeshMapperCuda3f_apply(size, maxNIn, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperMeshTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(size);
    MeshMapperCuda3f_apply(size, maxNIn, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperMeshTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    if (size == 0) return;
    calcMapT();
    MeshMapperCuda3f_applyPEq(insize, maxNOut, mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
}

template<>
void BarycentricMapperMeshTopology<CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::MatrixDeriv& out, const Out::MatrixDeriv& in)
{

    Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for ( Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();
        Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();

        int indexIn;

        if (colIt != colItEnd)
        {
            In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
            for ( ; colIt != colItEnd; ++colIt)
            {
                indexIn = colIt.index();
                InDeriv data = (InDeriv) colIt.val();

                for (int j=0; j<maxNIn; ++j)
                {
                    const OutReal f = ( OutReal ) getMapValue(indexIn,j);
                    int index = getMapIndex(indexIn,j);
                    if (index < 0) break;
                    o.addCol( index, data * f );
                }

            }
        }
    }
}

template<>
void BarycentricMapperMeshTopology<CudaVec3fTypes,CudaVec3fTypes>::draw (const core::visual::VisualParams* ,const Out::VecCoord& /*out*/, const In::VecCoord& /*in*/)
{
}

template<>
void BarycentricMapperMeshTopology<CudaVec3fTypes,CudaVec3fTypes>::resize( core::State<Out>* /*toModel*/ )
{
//    toModel->resize(size);
}





////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// Spread the instanciations over multiple files for more efficient and lightweight compilation

// Instantiations involving only CudaVec3fTypes
template class BarycentricMapping< CudaVec3fTypes, CudaVec3fTypes>;
template class BarycentricMapping< CudaVec3fTypes, ExtVec3fTypes>;


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

SOFA_DECL_CLASS(CudaBarycentricMapping_3f)

int BarycentricMappingCudaClass_3f = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< BarycentricMapping< CudaVec3fTypes, CudaVec3fTypes> >()
        .add< BarycentricMapping< CudaVec3fTypes, ExtVec3fTypes> >()
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
