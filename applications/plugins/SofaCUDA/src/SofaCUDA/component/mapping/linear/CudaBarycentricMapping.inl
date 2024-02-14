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
#pragma once

#include <SofaCUDA/component/mapping/linear/CudaBarycentricMapping.h>
#include <sofa/core/Mapping.inl>
#include <sofa/component/mapping/linear/BarycentricMapping.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/TopologyBarycentricMapper.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapper.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTopologyContainer.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperRegularGridTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperSparseGridTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperMeshTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperQuadSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTriangleSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperEdgeSetTopology.inl>
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl>
#include <sofa/component/topology/container/constant/MeshTopology.h>

namespace sofa::gpu::cuda
{

extern "C"
{
    void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void RegularGridMapperCuda3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void RegularGridMapperCuda3f_3f1_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_3f1_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void RegularGridMapperCuda3f1_3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);

    void SparseGridMapperCuda3f_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void SparseGridMapperCuda3f1_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void SparseGridMapperCuda3f_3f1_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_3f1_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f_3f1_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void SparseGridMapperCuda3f1_3f_apply(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_3f_applyJ(unsigned int size, const void * cudaHexa,const void* map, void* out, const void* in);
    void SparseGridMapperCuda3f1_3f_applyJT(unsigned int size, const void * CudaTnb, const void * CudaTst, const void * CudaTid, const void * CudaTVal, void* out, const void* in);

    void MeshMapperCuda3f_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f_3f1_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_3f_apply(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);

    void MeshMapperCuda3f_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f_3f1_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
    void MeshMapperCuda3f1_3f_applyPEq(unsigned int size, unsigned int maxN, const void* map, void* out, const void* in);
}

} // namespace sofa::gpu::cuda

namespace sofa::component::mapping::linear
{

using namespace gpu::cuda;


////////////////////////////////////////////////////////////
//////////          RegularGridTopology           //////////
////////////////////////////////////////////////////////////


template <typename VecIn, typename VecOut>
void BarycentricMapperRegularGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::clear(std::size_t reserve)
{
    map.clear();
    if (reserve>0) map.reserve(reserve);
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperRegularGridTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperRegularGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInCube(const Index cubeIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    CubeData& data = map[map.size()-1];
    //data.in_index = cubeIndex;
    data.in_index = topology->getHexaCopy(cubeIndex)[0];

    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    maxNOut = 0; // mapT must be recomputed
    return map.size()-1;
}

template <typename VecIn, typename VecOut>
void BarycentricMapperRegularGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::init(const typename Out::VecCoord& out, const typename In::VecCoord& /*in*/)
{
    int outside = 0;

    clear(out.size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        type::Vec3 coefs;
        int cube = topology->findCube(type::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        if (cube==-1)
        {
            ++outside;
            cube = topology->findNearestCube(type::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        }
        type::Vec<3,SReal> baryCoords = coefs;
        addPointInCube(cube, baryCoords.ptr());
    }
}

template <typename VecIn, typename VecOut>
void BarycentricMapperRegularGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::calcMapT()
{
    if (!map.empty() && maxNOut == 0)
    {
        const Index insize = topology->getNbPoints();
        const unsigned int gridsize[3] = { (unsigned int)topology->getNx(), (unsigned int)topology->getNy(), (unsigned int)topology->getNz() };
        // compute mapT
        const int nx = gridsize[0];
        const int nxny = gridsize[0]*gridsize[1];
        const int shift[8] = { 0, 1, nx, 1+nx, nxny, 1+nxny, nx+nxny, 1+nx+nxny };
        std::vector<int> nout(insize);
        for (std::size_t i=0; i<map.size(); i++)
        {
            const Index index0 = map[i].in_index;
            for (int j=0; j<8; j++)
                nout[index0+shift[j]]++;
        }
        for (Index i=0; i<insize; i++)
            if (Index(nout[i]) > maxNOut) maxNOut = nout[i];
        const int nbloc = (insize+BSIZE-1)/BSIZE;
        msg_info() << "CudaBarycentricMapping: mapT with "<<maxNOut<<" entries per DOF and "<<nbloc<<" blocs.";
        mapT.resize(nbloc*(BSIZE*maxNOut));
        for (unsigned int i=0; i<mapT.size(); i++)
            mapT[i] = std::make_pair(-1,0.0f);
        nout.clear();
        nout.resize(insize);
        for (unsigned int i=0; i<map.size(); i++)
        {
            const Index index0 = map[i].in_index;
            for (int j=0; j<8; j++)
            {
                int index = index0+shift[j];
                const int num = nout[index]++;
                const int b = (index / BSIZE); index -= b*BSIZE;
                float f;
                f  = (j&1)?(map[i].baryCoords[0]):(1-map[i].baryCoords[0]);
                f *= (j&2)?(map[i].baryCoords[1]):(1-map[i].baryCoords[1]);
                f *= (j&4)?(map[i].baryCoords[2]):(1-map[i].baryCoords[2]);
                //std::cout << "mapT["<<b<<"*"<<maxNOut*BSIZE<<"+"<<num<<"*"<<BSIZE<<"+"<<index<<"] = < "<<i<<", "<<f<<">"<<std::endl;
                mapT[(maxNOut*b+num)*BSIZE+index] = std::make_pair(i,f);
            }
        }
    }
}






////////////////////////////////////////////////////////////////////////////
//////////          BarycentricMapperSparseGridTopology           //////////
////////////////////////////////////////////////////////////////////////////


template <typename VecIn, typename VecOut>
void BarycentricMapperSparseGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::clear(std::size_t reserve)
{
    map.clear();
    bHexa = true;
    bTrans = true;
    if (reserve>0) map.reserve(reserve);
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperSparseGridTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperSparseGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInCube(const Index cubeIndex, const SReal* baryCoords)
{
    map.resize(map.size()+1);
    CubeData& data = map[map.size()-1];

    data.in_index = cubeIndex;
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    bHexa = true;
    bTrans = true;
    return map.size()-1;
}

template <typename VecIn, typename VecOut>
void BarycentricMapperSparseGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::init(const typename Out::VecCoord& out, const typename In::VecCoord& /*in*/)
{
    int outside = 0;

    clear(out.size());
    for (unsigned int i=0; i<out.size(); i++)
    {
        type::Vec3 coefs;
        int cube = topology->findCube(type::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        if (cube==-1)
        {
            ++outside;
            cube = topology->findNearestCube(type::Vec3(out[i]), coefs[0], coefs[1], coefs[2]);
        }
        type::Vec<3,SReal> baryCoords = coefs;
        addPointInCube(cube, baryCoords.ptr());
    }

    bHexa = true;
    bTrans = true;
}

template <typename VecIn, typename VecOut>
void BarycentricMapperSparseGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::buildHexa()
{
    if (! bHexa) return;

    CudaHexa.clear();

    const sofa::type::vector<Hexahedron>& hexaArray = this->topology->getHexahedra();
    for (unsigned i = 0; i < hexaArray.size(); i++)
    {
        const Hexahedron& cube = hexaArray[i];
        for (int c = 0; c < 8; c++) {
            CudaHexa.push_back(cube[c]);
        }
    }

    bHexa = false;
}

template <typename VecIn, typename VecOut>
void BarycentricMapperSparseGridTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::buildTranslate(unsigned outsize)
{
    if (! bTrans) return;

    CudaTnb.clear();
    CudaTst.clear();
    CudaTnb.resize(outsize);
    CudaTst.resize(outsize+1);

    for (unsigned i=0; i<outsize; i++) CudaTnb[i] = 0;

    for (unsigned i=0; i<map.size(); i++)
    {
        const auto cube = this->topology->getHexahedron(map[i].in_index);
        for (int c=0; c<8; c++) CudaTnb[cube[c]]++;
    }

    CudaTst[0] = 0;
    for (unsigned i=1; i<=outsize; i++)
    {
        CudaTst[i] = CudaTst[i-1] + CudaTnb[i-1];
        CudaTnb[i-1] = 0; //clear tnb
    }

    CudaTid.clear();
    CudaTVal.clear();
    CudaTid.resize(CudaTst[outsize]);
    CudaTVal.resize(CudaTst[outsize]);

    for (unsigned i=0; i<map.size(); i++)
    {
        const auto cube = this->topology->getHexahedron(map[i].in_index);

        const Real fx = map[i].baryCoords[0];
        const Real fy = map[i].baryCoords[1];
        const Real fz = map[i].baryCoords[2];

        for (int c=0; c<8; c++)
        {
            const int writepos = CudaTst[cube[c]] + CudaTnb[cube[c]];

            CudaTid[writepos] = i;

            if (c==0)      CudaTVal[writepos] = ( ( 1-fx ) * ( 1-fy ) * ( 1-fz ) );
            else if (c==1) CudaTVal[writepos] = ( (   fx ) * ( 1-fy ) * ( 1-fz ) );
            else if (c==3) CudaTVal[writepos] = ( ( 1-fx ) * (   fy ) * ( 1-fz ) );
            else if (c==2) CudaTVal[writepos] = ( (   fx ) * (   fy ) * ( 1-fz ) );
            else if (c==4) CudaTVal[writepos] = ( ( 1-fx ) * ( 1-fy ) * (   fz ) );
            else if (c==5) CudaTVal[writepos] = ( (   fx ) * ( 1-fy ) * (   fz ) );
            else if (c==7) CudaTVal[writepos] = ( ( 1-fx ) * (   fy ) * (   fz ) );
            else if (c==6) CudaTVal[writepos] = ( (   fx ) * (   fy ) * (   fz ) );

            CudaTnb[cube[c]]++;
        }
    }

    sizeout = outsize;
    bTrans = false;
}






////////////////////////////////////////////////////////////
//////////            BaseMeshTopology            //////////
////////////////////////////////////////////////////////////


template <typename VecIn, typename VecOut>
void BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::resizeMap(std::size_t size2, std::size_t maxNIn2)
{
    if (maxNIn2 < maxNIn) maxNIn2 = maxNIn;
    map.resize(((size2+BSIZE-1)/BSIZE)*maxNIn2);
    if (maxNIn2 > maxNIn)
    {
        const int n = (size+BSIZE-1)/BSIZE;
        for (int b=n-1; b>0; --b)
            for (int j=maxNIn-1; j>=0; --j)
            {
                // copy old data
                map[b*maxNIn2+j]=map[b*maxNIn+j];
                // clear other data
                for (int i=0; i<BSIZE; ++i)
                {
                    map[b*maxNIn+j].d[i].i = 0;
                    map[b*maxNIn+j].d[i].val = 0.0f;
                }
            }
    }
    size = size2;
    maxNIn = maxNIn2;
}

template <typename VecIn, typename VecOut>
void BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::setMap(Index outIndex, Index j, Index inIndex, Real val)
{
    const int b    = outIndex / BSIZE;
    outIndex = outIndex % BSIZE;
    map[b*maxNIn+j].d[outIndex].i = inIndex+1;
    map[b*maxNIn+j].d[outIndex].val = val;
}

template <typename VecIn, typename VecOut>
float BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::getMapValue(Index outIndex, Index j)
{
    const int b    = outIndex / BSIZE;
    outIndex = outIndex % BSIZE;
    return map[b*maxNIn+j].d[outIndex].val;
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::getMapIndex(Index outIndex, Index j)
{
    const int b    = outIndex / BSIZE;
    outIndex = outIndex % BSIZE;
    return map[b*maxNIn+j].d[outIndex].i-1;
}

template <typename VecIn, typename VecOut>
void BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::clear(std::size_t reserve)
{
    map.clear(); if (reserve>0) map.reserve((reserve+BSIZE-1)/BSIZE*maxNIn);
    size = 0;
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInLine(const Index lineIndex, const SReal* baryCoords)
{
    const unsigned int i0 = size;
    resizeMap(i0+1,2);
    core::topology::BaseMeshTopology::Line e = topology->getLine(lineIndex);
    setMap(i0,0,e[0],(Real)(1-baryCoords[0]));
    setMap(i0,1,e[1],(Real)(baryCoords[0]));
    return i0;
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInTriangle(const Index triangleIndex, const SReal* baryCoords)
{
    const unsigned int i0 = size;
    resizeMap(i0+1,3);
    core::topology::BaseMeshTopology::Triangle e = topology->getTriangle(triangleIndex);
    setMap(i0,0,e[0],(Real)(1-baryCoords[0]-baryCoords[1]));
    setMap(i0,1,e[1],(Real)(baryCoords[0]));
    setMap(i0,2,e[2],(Real)(baryCoords[1]));
    return i0;
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInQuad(const Index quadIndex, const SReal* baryCoords)
{
    const unsigned int i0 = size;
    resizeMap(i0+1,4);
    core::topology::BaseMeshTopology::Quad e = topology->getQuad(quadIndex);
    setMap(i0,0,e[0],(Real)((1-baryCoords[0])*(1-baryCoords[1])));
    setMap(i0,1,e[1],(Real)((  baryCoords[0])*(1-baryCoords[1])));
    setMap(i0,2,e[3],(Real)((1-baryCoords[0])*(  baryCoords[1])));
    setMap(i0,3,e[2],(Real)((  baryCoords[0])*(  baryCoords[1])));
    return i0;
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInTetra(const Index tetraIndex, const SReal* baryCoords)
{
    const unsigned int i0 = size;
    resizeMap(i0+1,4);
    core::topology::BaseMeshTopology::Tetra e = topology->getTetrahedron(tetraIndex);
    setMap(i0,0,e[0],(Real)(1-baryCoords[0]-baryCoords[1]-baryCoords[2]));
    setMap(i0,1,e[1],(Real)(baryCoords[0]));
    setMap(i0,2,e[2],(Real)(baryCoords[1]));
    setMap(i0,3,e[3],(Real)(baryCoords[2]));
    return i0;
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::addPointInCube(const Index cubeIndex, const SReal* baryCoords)
{
    const unsigned int i0 = size;
    resizeMap(i0+1,8);

    core::topology::BaseMeshTopology::Hexa e = topology->getHexahedron(cubeIndex);
    setMap(i0,0,e[0],(Real)((1-baryCoords[0])*(1-baryCoords[1])*(1-baryCoords[2])));
    setMap(i0,1,e[1],(Real)((  baryCoords[0])*(1-baryCoords[1])*(1-baryCoords[2])));
    setMap(i0,2,e[3],(Real)((1-baryCoords[0])*(  baryCoords[1])*(1-baryCoords[2])));
    setMap(i0,3,e[2],(Real)((  baryCoords[0])*(  baryCoords[1])*(1-baryCoords[2])));
    setMap(i0,4,e[4],(Real)((1-baryCoords[0])*(1-baryCoords[1])*(  baryCoords[2])));
    setMap(i0,5,e[5],(Real)((  baryCoords[0])*(1-baryCoords[1])*(  baryCoords[2])));
    setMap(i0,6,e[7],(Real)((1-baryCoords[0])*(  baryCoords[1])*(  baryCoords[2])));
    setMap(i0,7,e[6],(Real)((  baryCoords[0])*(  baryCoords[1])*(  baryCoords[2])));

    return i0;
}


template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::createPointInLine(const typename Out::Coord& p, Index lineIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[1];
    const auto& elem = topology->getLine(lineIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    typename In::Coord pos = p - p0;
    baryCoords[0] = ((pos*pA)/pA.norm2());
    return this->addPointInLine(lineIndex, baryCoords);
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::createPointInTriangle(const typename Out::Coord& p, Index triangleIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[2];
    const auto& elem = topology->getTriangle(triangleIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    const typename In::Coord pB = (*points)[elem[2]] - p0;
    typename In::Coord pos = p - p0;
    // First project to plane
    typename In::Coord normal = cross(pA, pB);
    Real norm2 = normal.norm2();
    pos -= normal*((pos*normal)/norm2);
    baryCoords[0] = (Real)sqrt(cross(pB, pos).norm2() / norm2);
    baryCoords[1] = (Real)sqrt(cross(pA, pos).norm2() / norm2);
    return this->addPointInTriangle(triangleIndex, baryCoords);
}

template <typename VecIn, typename VecOut>
typename BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn, VecIn, float>, gpu::cuda::CudaVectorTypes<VecOut, VecOut, float> >::Index
BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::createPointInQuad(const typename Out::Coord& p, Index quadIndex, const typename In::VecCoord* points)
{
    SReal baryCoords[2];
    const auto& elem = topology->getQuad(quadIndex);
    const typename In::Coord p0 = (*points)[elem[0]];
    const typename In::Coord pA = (*points)[elem[1]] - p0;
    const typename In::Coord pB = (*points)[elem[3]] - p0;
    typename In::Coord pos = p - p0;
    type::Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross(pA, pB);
    mt.transpose(m);
    base.invert(mt);
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad(quadIndex, baryCoords);
}


template <typename VecIn, typename VecOut>
void BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::init(const typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    int outside = 0;

    const auto& tetras = topology->getTetrahedra();
    const auto& cubes = topology->getHexahedra();

    const auto& triangles = topology->getTriangles();
    const auto& quads = topology->getQuads();
    sofa::type::vector<type::Matrix3> bases;
    sofa::type::vector<type::Vec3> centers;
    clear(out.size()); // reserve space for mapping
    if (tetras.empty() && cubes.empty())
    {
        // no 3D elements -> map on 2D elements
        const int c0 = triangles.size();
        bases.resize(triangles.size()+quads.size());
        centers.resize(triangles.size()+quads.size());
        for (unsigned int t = 0; t < triangles.size(); t++)
        {
            type::Mat3x3d m,mt;
            m[0] = in[triangles[t][1]]-in[triangles[t][0]];
            m[1] = in[triangles[t][2]]-in[triangles[t][0]];
            m[2] = cross(m[0],m[1]);
            mt.transpose(m);
            bases[t].invert(mt);
            centers[t] = (in[triangles[t][0]]+in[triangles[t][1]]+in[triangles[t][2]])/3;
        }
        for (unsigned int c = 0; c < quads.size(); c++)
        {
            type::Mat3x3d m,mt;
            m[0] = in[quads[c][1]]-in[quads[c][0]];
            m[1] = in[quads[c][3]]-in[quads[c][0]];
            m[2] = cross(m[0],m[1]);
            mt.transpose(m);
            bases[c0+c].invert(mt);
            centers[c0+c] = (in[quads[c][0]]+in[quads[c][1]]+in[quads[c][2]]+in[quads[c][3]])*0.25;
        }
        for (unsigned int i=0; i<out.size(); i++)
        {
            type::Vec3 pos = out[i];
            type::Vec3 coefs;
            int index = -1;
            double distance = 1e10;
            for (unsigned int t = 0; t < triangles.size(); t++)
            {
                type::Vec3d v = bases[t] * (pos - in[triangles[t][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max((v[2]<0?-v[2]:v[2])-0.01,v[0]+v[1]-1));
                if (d>0) d = (pos-centers[t]).norm2();
                if (d<distance) { coefs = v; distance = d; index = t; }
            }
            for (unsigned int c = 0; c < quads.size(); c++)
            {
                type::Vec3d v = bases[c0+c] * (pos - in[quads[c][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(v[1]-1,v[0]-1),std::max(v[2]-0.01,-v[2]-0.01)));
                if (d>0) d = (pos-centers[c0+c]).norm2();
                if (d<distance) { coefs = v; distance = d; index = c0+c; }
            }
            if (distance>0)
            {
                ++outside;
            }
            if (index < c0)
                addPointInTriangle(index, coefs.ptr());
            else
                addPointInQuad(index-c0, coefs.ptr());
        }
    }
    else
    {
        const int c0 = tetras.size();
        bases.resize(tetras.size()+cubes.size());
        centers.resize(tetras.size()+cubes.size());
        for (unsigned int t = 0; t < tetras.size(); t++)
        {
            type::Mat3x3d m,mt;
            m[0] = in[tetras[t][1]]-in[tetras[t][0]];
            m[1] = in[tetras[t][2]]-in[tetras[t][0]];
            m[2] = in[tetras[t][3]]-in[tetras[t][0]];
            mt.transpose(m);
            bases[t].invert(mt);
            centers[t] = (in[tetras[t][0]]+in[tetras[t][1]]+in[tetras[t][2]]+in[tetras[t][3]])*0.25;
            //std::cout << "Tetra "<<t<<" center="<<centers[t]<<" base="<<m<<std::endl;
        }
        for (unsigned int c = 0; c < cubes.size(); c++)
        {
            type::Mat3x3d m,mt;
            m[0] = in[cubes[c][1]]-in[cubes[c][0]];
            m[1] = in[cubes[c][3]]-in[cubes[c][0]];
            m[2] = in[cubes[c][4]]-in[cubes[c][0]];
            mt.transpose(m);
            bases[c0+c].invert(mt);
            centers[c0+c] = (in[cubes[c][0]]+in[cubes[c][1]]+in[cubes[c][2]]+in[cubes[c][3]]+in[cubes[c][4]]+in[cubes[c][5]]+in[cubes[c][6]]+in[cubes[c][7]])*0.125;
        }
        for (unsigned int i=0; i<out.size(); i++)
        {
            type::Vec3 pos = out[i];
            type::Vec3 coefs;
            int index = -1;
            double distance = 1e10;
            for (unsigned int t = 0; t < tetras.size(); t++)
            {
                type::Vec3 v = bases[t] * (pos - in[tetras[t][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max(-v[2],v[0]+v[1]+v[2]-1));
                if (d>0) d = (pos-centers[t]).norm2();
                if (d<distance) { coefs = v; distance = d; index = t; }
            }
            for (unsigned int c = 0; c < cubes.size(); c++)
            {
                type::Vec3 v = bases[c0+c] * (pos - in[cubes[c][0]]);
                double d = std::max(std::max(-v[0],-v[1]),std::max(std::max(-v[2],v[0]-1),std::max(v[1]-1,v[2]-1)));
                if (d>0) d = (pos-centers[c0+c]).norm2();
                if (d<distance) { coefs = v; distance = d; index = c0+c; }
            }
            if (distance>0)
            {
                ++outside;
            }
            if (index < c0)
                addPointInTetra(index, coefs.ptr());
            else
                addPointInCube(index-c0, coefs.ptr());
        }
    }
    msg_info() << "CUDA: BarycentricMapperMeshTopology: map initialized, "<<size<<" output points, " << outside << " points ouside input mesh, max "<<maxNIn<<" inputs points per output, "<<map.size()*BSIZE<<" contributions total.";
}


template <typename VecIn, typename VecOut>
void BarycentricMapperMeshTopology<gpu::cuda::CudaVectorTypes<VecIn,VecIn,float>, gpu::cuda::CudaVectorTypes<VecOut,VecOut,float> >::calcMapT()
{
    if (size > 0 && maxNOut == 0)
    {
        // compute mapT
        std::vector<int> nout;
        const int nb = (size+BSIZE-1)/BSIZE;
        for (int b=0; b<nb; b++)
        {
            for (Index j=0; j<Index(maxNIn); j++)
            {
                const int n = (b<nb-1) ? BSIZE : Index(size-b*BSIZE);
                for (int i=0; i<n; i++)
                {
                    const int index = map[b*maxNIn+j].d[i].i-1;
                    //std::cout << "map["<<b<<"*"<<maxNIn<<"+"<<j<<"].index["<<i<<"]="<<index<<std::endl;
                    if (index >= 0)
                    {
                        if ((unsigned)index >= nout.size()) nout.resize(index+1);
                        nout[index]++;
                    }
                }
            }
        }
        insize = nout.size();
        for (Index i=0; i<Index(insize); i++)
            if (Index(nout[i]) > maxNOut) maxNOut = nout[i];
        const int nbloc = (insize+BSIZE-1)/BSIZE;
        msg_info() << "CudaBarycentricMapping: mapT with "<<maxNOut<<" entries per DOF and "<<nbloc<<" blocs.";
        mapT.clear();
        mapT.resize(nbloc*maxNOut);
        //for (unsigned int i=0;i<mapT.size();i++)
        //    mapT[i] = std::make_pair(-1,0.0f);
        nout.clear();
        nout.resize(insize);
        for (int b=0; b<nb; b++)
        {
            for (Index j=0; j<Index(maxNIn); j++)
            {
                const int n = (b<nb-1) ? BSIZE : Index(size-b*BSIZE);
                for (int i=0; i<n; i++)
                {
                    int index = map[b*maxNIn+j].d[i].i-1;
                    float val = (float) map[b*maxNIn+j].d[i].val;
                    if (index >= 0)
                    {
                        const int num = nout[index]++;
                        const int bo = (index / BSIZE); index -= bo*BSIZE;
                        mapT[bo*maxNOut+num].d[index].i = b*BSIZE+i;
                        mapT[bo*maxNOut+num].d[index].val = val;
                    }
                }
            }
        }
    }
}

} // namespace sofa::component::mapping::linear

