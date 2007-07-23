#ifndef SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_INL
#define SOFA_GPU_CUDA_CUDABARYCENTRICMAPPING_INL

#include "CudaBarycentricMapping.h"
#include <sofa/component/mapping/BarycentricMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void RegularGridMapperCuda3f_apply(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJ(unsigned int size, const unsigned int* gridsize, const void* map, void* out, const void* in);
    void RegularGridMapperCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const unsigned int* gridsize, const void* mapT, void* out, const void* in);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;


void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::clear(int reserve)
{
    map.clear();
    if (reserve>0) map.reserve(reserve);
}

int TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::addPointInCube(int cubeIndex, const Real* baryCoords)
{
    map.resize(map.size()+1);
    CubeData& data = map[map.size()-1];
    //data.in_index = cubeIndex;
    data.in_index = topology->getCube(cubeIndex)[0];
    data.baryCoords[0] = baryCoords[0];
    data.baryCoords[1] = baryCoords[1];
    data.baryCoords[2] = baryCoords[2];
    maxNOut = 0; // mapT must be recomputed
    return map.size()-1;
}

void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::init()
{
}

void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    unsigned int gridsize[3] = { topology->getNx(), topology->getNy(), topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f_apply(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    unsigned int gridsize[3] = { topology->getNx(), topology->getNy(), topology->getNz() };
    out.fastResize(map.size());
    RegularGridMapperCuda3f_applyJ(map.size(), gridsize, map.deviceRead(), out.deviceWrite(), in.deviceRead());
}

void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    unsigned int gridsize[3] = { topology->getNx(), topology->getNy(), topology->getNz() };

    if (map.size() == 0) return;
    unsigned int insize = out.size();
    if (maxNOut == 0)
    {
        // compute mapT
        const int nx = gridsize[0];
        const int nxny = gridsize[0]*gridsize[1];
        const int shift[8] = { 0, 1, nx, 1+nx, nxny, 1+nxny, nx+nxny, 1+nx+nxny };
        std::vector<int> nout(insize);
        for (unsigned int i=0; i<map.size(); i++)
        {
            int index0 = map[i].in_index;
            for (int j=0; j<8; j++)
                nout[index0+shift[j]]++;
        }
        for (unsigned int i=0; i<insize; i++)
            if (nout[i] > maxNOut) maxNOut = nout[i];
        int nbloc = (insize+BSIZE-1)/BSIZE;
        std::cout << "CudaBarycentricMapping: mapT with "<<maxNOut<<" entries per DOF and "<<nbloc<<" blocs."<<std::endl;
        mapT.resize(nbloc*(BSIZE*maxNOut));
        for (unsigned int i=0; i<mapT.size(); i++)
            mapT[i] = std::make_pair(-1,0.0f);
        nout.clear();
        nout.resize(insize);
        for (unsigned int i=0; i<map.size(); i++)
        {
            int index0 = map[i].in_index;
            for (int j=0; j<8; j++)
            {
                int index = index0+shift[j];
                int num = nout[index]++;
                int b = (index / BSIZE); index -= b*BSIZE;
                float f;
                f  = (j&1)?(map[i].baryCoords[0]):(1-map[i].baryCoords[0]);
                f *= (j&2)?(map[i].baryCoords[1]):(1-map[i].baryCoords[1]);
                f *= (j&4)?(map[i].baryCoords[2]):(1-map[i].baryCoords[2]);
                std::cout << "mapT["<<b<<"*"<<maxNOut*BSIZE<<"+"<<num<<"*"<<BSIZE<<"+"<<index<<"] = < "<<i<<", "<<f<<">"<<std::endl;
                mapT[(maxNOut*b+num)*BSIZE+index] = std::make_pair(i,f);
            }
        }
    }


    RegularGridMapperCuda3f_applyJT(insize, maxNOut, gridsize, mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
}

void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::applyJT( In::VecConst& out, const Out::VecConst& in )
{
}

void TopologyBarycentricMapper<topology::RegularGridTopology,CudaVec3fTypes,CudaVec3fTypes>::draw( const Out::VecCoord& out, const In::VecCoord& in)
{
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
