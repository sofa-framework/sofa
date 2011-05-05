/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/container/SpatialGridContainer.inl>
#include <sofa/helper/gl/template.h>

#ifdef SOFA_GPU_CUDPP
#include <cudpp.h>
#include <cudpp_plan.h>
#include <cudpp_plan_manager.h>
#include <cudpp_radixsort.h>
#else
#ifdef SOFA_DEV
#include "radixsort.h"
#endif
#endif

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
#ifdef SOFA_GPU_CUDPP
    cudppHandleSortMaxElements = 0;
#endif
}

template<class TCoord, class TDeriv, class TReal>
SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::~SpatialGrid()
{
#ifdef SOFA_GPU_CUDPP
    if (cudppHandleSortMaxElements > 0)
        cudppDestroyPlan(cudppHandleSort);
#endif
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
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3fTypes > >::kernel_updateGrid(
    int cellBits, int index0, Real cellWidth, int nbPoints, void* particleIndex, void* particleHash,
#ifndef SOFA_GPU_CUDPP
    void* sortTmp,
#endif
    void* cells, void* cellGhost, const void* x)
{
    gpu::cuda::SpatialGridContainer3f_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);

    int nbbits = 8;
    while (nbbits < cellBits + 1) nbbits+=8;

#ifdef SOFA_GPU_CUDPP
    //std::cout << "ERROR : CUDPP is not compatible with SpatialGrid\n";
    cudppSort(cudppHandleSort,particleHash,particleIndex,nbbits,nbPoints*8);
#else
#ifdef SOFA_DEV
    radixSort((unsigned int *)particleHash, (unsigned int *)particleIndex, (unsigned int *)sortTmp /*.deviceWrite()*/, nbPoints*8, nbbits);
#else
    std::cout << "ERROR : CUDPP is required for SpatialGrid\n";
#endif
#endif

    gpu::cuda::SpatialGridContainer_findCellRange(cellBits, index0, cellWidth, nbPoints, particleHash, cells, cellGhost);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3f1Types > >::kernel_updateGrid(
    int cellBits, int index0, Real cellWidth, int nbPoints, void* particleIndex, void* particleHash,
#ifndef SOFA_GPU_CUDPP
    void* sortTmp,
#endif
    void* cells, void* cellGhost, const void* x)
{
    gpu::cuda::SpatialGridContainer3f1_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);

    int nbbits = 8;
    while (nbbits < cellBits + 1) nbbits+=8;

#ifdef SOFA_GPU_CUDPP
    //std::cout << "ERROR : CUDPP is not compatible with SpatialGrid\n";
    cudppSort(cudppHandleSort,particleHash,particleIndex,nbbits,nbPoints*8);
#else
#ifdef SOFA_DEV
    radixSort((unsigned int *)particleHash, (unsigned int *)particleIndex, (unsigned int *)sortTmp, nbPoints*8, nbbits);
#else
    std::cout << "ERROR : CUDPP is required for SpatialGrid\n";
#endif
#endif

    gpu::cuda::SpatialGridContainer_findCellRange(cellBits, index0, cellWidth, nbPoints, particleHash, cells, cellGhost);
}

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3dTypes > >::kernel_updateGrid(
    int /*cellBits*/, int /*index0*/, Real /*cellWidth*/, int /*nbPoints*/, void* /*particleIndex*/, void* /*particleHash*/,
#ifndef SOFA_GPU_CUDPP
    void* /*sortTmp*/,
#endif
    void* /*cells*/, void* /*cellGhost*/, const void* /*x*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3dTypes > >::kernel_updateGrid(int cellBits, int index0, Real cellWidth, int nbPoints, void* particleIndex, void* particleHash, void* sortTmp, void* cells, void* cellGhost, const void* x)"<<std::endl;
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

#ifdef SOFA_GPU_CUDPP
    unsigned int numElements = (unsigned int)nbPoints*8;
    if (numElements > cudppHandleSortMaxElements)
    {
        if (cudppHandleSortMaxElements > 0)
        {
            cudppDestroyPlan(cudppHandleSort);
            cudppHandleSortMaxElements = (((cudppHandleSortMaxElements>>7)+1)<<7); // increase size to at least the next multiple of 128
        }
        if (numElements > cudppHandleSortMaxElements)
            cudppHandleSortMaxElements = numElements;
        cudppHandleSortMaxElements = ((cudppHandleSortMaxElements + 255) & ~255);

        std::cout << "Creating CUDPP RadixSort Plan for " << cudppHandleSortMaxElements << " elements." << std::endl;
        CUDPPConfiguration config;
        config.algorithm = CUDPP_SORT_RADIX;
        config.datatype = CUDPP_UINT;
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
        if (cudppPlan(&cudppHandleSort, config, cudppHandleSortMaxElements, 1, 0) != CUDPP_SUCCESS)
        {
            std::cerr << "ERROR creating CUDPP RadixSort Plan for " << cudppHandleSortMaxElements << " elements." << std::endl;
            cudppHandleSortMaxElements = 0;
            cudppDestroyPlan(cudppHandleSort);
        }
    }
#else
#ifdef SOFA_DEV
    sortTmp.recreate(radixSortTempStorage(nbPoints*8));
    sortTmp.deviceWrite();
#endif
#endif
    //cells.recreate(nbCells+1);
    cellGhost.recreate(nbCells);
    //sortedPos.recreate(nbPoints);
    kernel_updateGrid(
        cellBits, index0, cellWidth*2, nbPoints, cells.deviceWriteAt(index0), particleHash.deviceWrite(),
#ifndef SOFA_GPU_CUDPP
        sortTmp.deviceWrite(),
#endif
        cells.deviceWrite(), cellGhost.deviceWrite(), x.deviceRead());

    std::cout << nbPoints*8 << " entries in " << nbCells << " cells." << std::endl;
    int nfill = 0;
    for (int c=0; c<nbCells; ++c)
    {
        if (cells[c] <= 0) continue;
        if (nfill >= 100 && nfill < 110)
            std::cout << "Cell " << c << ": range = " << cells[c]-index0 << " - " << (cells[c+1]&~(1U<<31))-index0 << "     ghost = " << cellGhost[c]-index0 << std::endl;
        ++nfill;
    }
    std::cout << ((1000*nfill)/nbCells) * 0.1 << " % cells with particles." << std::endl;

    //kernel_reorderData(nbPoints, particleHash.deviceRead(), sortedPos.deviceWrite(), x.deviceRead());
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::draw()
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
