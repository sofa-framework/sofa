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
//
// C++ Interface: SpatialGridContainer
//
// Description:
//
//
// Author: The SOFA team <http://www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_GPU_CUDA_CUDASPATIALGRIDCONTAINER_INL
#define SOFA_GPU_CUDA_CUDASPATIALGRIDCONTAINER_INL

#include <sofa/gpu/cuda/CudaSpatialGridContainer.h>
#include <sofa/gpu/cuda/CudaSort.h>
#include <SofaSphFluid/SpatialGridContainer.inl>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SpatialGridContainer3f_computeHash(int cellBits, float cellWidth, int nbPoints, void* particleIndex8, void* particleHash8, const void* x);
    void SpatialGridContainer3f1_computeHash(int cellBits, float cellWidth, int nbPoints, void* particleIndex8, void* particleHash8, const void* x);
    void SpatialGridContainer_findCellRange(int cellBits, int index0, float cellWidth, int nbPoints, const void* particleHash8, void* cellRange, void* cellGhost);
//void SpatialGridContainer3f_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x);
//void SpatialGridContainer3f1_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace container
{

using namespace sofa::helper;

//      template<class TCoord, class TDeriv, class TReal>
//      typename SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::Grid SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::emptyGrid;

template<class TCoord, class TDeriv, class TReal>
SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::SpatialGrid(Real cellWidth)
    : cellWidth(cellWidth), invCellWidth(1/cellWidth), lastX(NULL)
{
    cellBits = 15;
    nbCells = 1<<cellBits;
}

template<class TCoord, class TDeriv, class TReal>
SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::~SpatialGrid()
{
}

template<class TCoord, class TDeriv, class TReal> template<class NeighborListener>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::findNeighbors(NeighborListener* /*dest*/, Real /*dist*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::findNeighbors(NeighborListener* dest, Real dist)"<<std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::computeField(ParticleField* /*field*/, Real /*dist*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::computeField(ParticleField* field, Real dist)"<<std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::reorderIndices(helper::vector<unsigned int>* /*old2new*/, helper::vector<unsigned int>* /*new2old*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::reorderIndices(helper::vector<unsigned int>* old2new, helper::vector<unsigned int>* new2old)"<<std::endl;
}


template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3fTypes > >::kernel_computeHash(
    int cellBits, Real cellWidth, int nbPoints, void* particleIndex, void* particleHash,
    const void* x)
{
    /*
        {
            helper::ReadAccessor< sofa::gpu::cuda::CudaVector< unsigned int > > pparticleHash = this->particleHash;
            for (int i=0;i<nbPoints; i+= nbPoints/9+1)
            {
                std::cout << "Chash["<<i<<"] =" ;
                for (int x=0;x<8;++x) std::cout << "\t" << pparticleHash[i*8+x];
                std::cout<<std::endl;
            }
        }
    */
    gpu::cuda::SpatialGridContainer3f_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);
    /*
        this->particleHash.deviceWrite();
        {
            helper::ReadAccessor< sofa::gpu::cuda::CudaVector< unsigned int > > pparticleHash = this->particleHash;
            for (int i=0;i<nbPoints; i+= nbPoints/9+1)
            {
                std::cout << "Ghash["<<i<<"] =" ;
                for (int x=0;x<8;++x) std::cout << "\t" << pparticleHash[i*8+x];
                std::cout<<std::endl;
            }
        }
    */
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3fTypes > >::kernel_updateGrid(
    int cellBits, int index0, Real cellWidth, int nbPoints, const void* particleHash,
    void* cells, void* cellGhost)
{
    //int nbbits = 8;
    //while (nbbits < cellBits + 1) nbbits+=8;
    //CudaSort(particleHash,particleIndex,nbbits,nbPoints*8);
    gpu::cuda::SpatialGridContainer_findCellRange(cellBits, index0, cellWidth, nbPoints, particleHash, cells, cellGhost);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3f1Types > >::kernel_computeHash(
    int cellBits, Real cellWidth, int nbPoints, void* particleIndex, void* particleHash,
    const void* x)
{
    gpu::cuda::SpatialGridContainer3f1_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3f1Types > >::kernel_updateGrid(
    int cellBits, int index0, Real cellWidth, int nbPoints, const void* particleHash,
    void* cells, void* cellGhost)
{
    gpu::cuda::SpatialGridContainer_findCellRange(cellBits, index0, cellWidth, nbPoints, particleHash, cells, cellGhost);
}

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3dTypes > >::kernel_computeHash(
    int /*cellBits*/, Real /*cellWidth*/, int /*nbPoints*/, void* /*particleIndex*/, void* /*particleHash*/,
    const void* /*x*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3dTypes > >::kernel_computeHash()"<<std::endl;
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3dTypes > >::kernel_updateGrid(
    int /*cellBits*/, int /*index0*/, Real /*cellWidth*/, int /*nbPoints*/, const void* /*particleHash*/,
    void* /*cells*/, void* /*cellGhost*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3dTypes > >::kernel_updateGrid()"<<std::endl;
}

#endif // SOFA_GPU_CUDA_DOUBLE

/*
template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3fTypes > >::kernel_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x)
{
    gpu::cuda::SpatialGridContainer3f_reorderData(nbPoints, particleHash, sorted, x);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3f1Types > >::kernel_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x)
{
    gpu::cuda::SpatialGridContainer3f1_reorderData(nbPoints, particleHash, sorted, x);
}
*/

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::update(const VecCoord& x)
{
    lastX = &x;
    data.clear();
    int nbPoints = x.size();
    int index0 = nbCells+BSIZE;
    /*particleIndex*/ cells.recreate(index0+nbPoints*8,8*BSIZE);
    particleHash.recreate(nbPoints*8,8*BSIZE);

    {
        unsigned int numElements = (unsigned int)nbPoints*8;
        sofa::gpu::cuda::CudaSortPrepare(numElements);
    }

#ifdef SOFA_DEV
    //sortTmp.recreate(radixSortTempStorage(nbPoints*8));
    //sortTmp.deviceWrite();
#endif

    //cells.recreate(nbCells+1);
    cellGhost.recreate(nbCells);
    //sortedPos.recreate(nbPoints);
#if 0
    {
        helper::WriteAccessor< sofa::gpu::cuda::CudaVector< int > > pcells = cells;
        helper::WriteAccessor< sofa::gpu::cuda::CudaVector< unsigned int > > pparticleHash = particleHash;
        helper::ReadAccessor< VecCoord > px = x;
        const Real pcellWidth = cellWidth*2.0f;
        //const Real pinvCellWidth = 1.0f/pcellWidth;
        const int pcellMask = (1<<cellBits)-1;
        //const Real phalfCellWidth = pcellWidth*0.5f;
        const Real pinvHalfCellWidth = 2.0f/pcellWidth;
        for (int i=0; i<nbPoints; ++i)
        {
            Coord p = px[i];
            int hgpos_x,hgpos_y,hgpos_z;
            hgpos_x = helper::rfloor(p[0] * pinvHalfCellWidth);
            hgpos_y = helper::rfloor(p[1] * pinvHalfCellWidth);
            hgpos_z = helper::rfloor(p[2] * pinvHalfCellWidth);
            int halfcell = ((hgpos_x&1) + ((hgpos_y&1)<<1) + ((hgpos_z&1)<<2))^7;
            // compute the first cell to be influenced by the particle
            hgpos_x = (hgpos_x-1) >> 1;
            hgpos_y = (hgpos_y-1) >> 1;
            hgpos_z = (hgpos_z-1) >> 1;
            unsigned int hx = (HASH_PX*hgpos_x);
            unsigned int hy = (HASH_PY*hgpos_y);
            unsigned int hz = (HASH_PZ*hgpos_z);
            for (int x=0; x<8; ++x)
            {
                unsigned int h_x = hx; if (x&1) h_x += HASH_PX;
                unsigned int h_y = hy; if (x&2) h_y += HASH_PY;
                unsigned int h_z = hz; if (x&4) h_z += HASH_PZ;
                unsigned int hash = ((h_x ^ h_y ^ h_z) & pcellMask)<<1;
                if (halfcell != x) ++hash;
                pcells[index0 + i*8 + x] = i;
                pparticleHash[i*8 + x] = hash;
            }
        }
    }
#endif
#if 0
    helper::vector< std::pair<unsigned int,int> > cpusort;
    cpusort.resize(8*nbPoints);
    for (int i=0; i<8*nbPoints; ++i)
        cpusort[i] = std::make_pair(pparticleHash[i],pcells[index0+i]);
    std::sort(cpusort.begin(),cpusort.end(),compare_pair_first);

    for (int i=0; i<8*nbPoints; ++i)
    {
        pparticleHash[i] = cpusort[i].first;
        pcells[index0+i] = cpusort[i].second;
    }
#endif

    kernel_computeHash(
        cellBits, cellWidth*2, nbPoints, cells.deviceWriteAt(index0), particleHash.deviceWrite(),
        x.deviceRead());

    {
        int nbbits = 8;
        while (nbbits < cellBits + 1) nbbits+=8;
        sofa::gpu::cuda::CudaSort(&particleHash,0, &cells,index0, nbPoints*8, nbbits);
    }

    kernel_updateGrid(
        cellBits, index0, cellWidth*2, nbPoints, particleHash.deviceRead(),
        cells.deviceWrite(), cellGhost.deviceWrite());
#if 0
    std::cout << nbPoints*8 << " entries in " << nbCells << " cells." << std::endl;
    int nfill = 0;
    for (int c=0; c<nbCells; ++c)
    {
        int cellBegin = cells[c];
        int cellEnd = cells[c+1]&~(1U<<31);
        if (cells[c] <= 0) continue;
        if (cellEnd > cellBegin + nbPoints/2) // || nfill >= 100 && nfill < 110)
            std::cout << "Cell " << c << ": range = " << cellBegin-index0 << " - " << cellEnd-index0 << "     ghost = " << cellGhost[c]-index0 << std::endl;
        ++nfill;
    }
    std::cout << ((1000*nfill)/nbCells) * 0.1 << " % cells with particles." << std::endl;
#endif

    //kernel_reorderData(nbPoints, particleHash.deviceRead(), sortedPos.deviceWrite(), x.deviceRead());
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::draw(const core::visual::VisualParams* )
{
    if (!lastX) return;
    int nbPoints = particleHash.size();
    int index0 = nbCells+BSIZE;
    glDisable(GL_LIGHTING);
    glColor4f(1,1,1,1);
    glPointSize(3);
    glBegin(GL_POINTS);
    unsigned int last = 0;
    for (int i=0; i<nbPoints; i++)
    {
        unsigned int cell = particleHash[i];
        unsigned int p = cells[index0+i]; //particleIndex[i];
        if (cell < last)
        {
            std::cerr << "SORT ERROR: index " << i << " key " << cell << " value " << p << " last key " << last << std::endl;
        }
        last = cell;
        if (!(cell&1)) continue; // this is a ghost particle from a neighbor cell
        cell>>=1;
        //if (cell != 0 && cell != 65535)
        //    std::cout << i << ": "<<p<<" -> "<<cell<<", "<<(*lastX)[p]<<" -> "<<sortedPos[i]<<std::endl;
        int r = cell&3;
        int g = (cell>>2)&3;
        int b = (cell>>4)&3;
        glColor4ub(63+r*64,63+g*64,63+b*64,255);
        //glVertex3fv(sortedPos[i].ptr());
        helper::gl::glVertexT((*lastX)[p]);
    }
    glEnd();
    glPointSize(1);
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
