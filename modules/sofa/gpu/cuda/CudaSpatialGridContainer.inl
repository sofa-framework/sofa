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

#ifdef SOFA_DEV
#include "radixsort.h"
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
    void SpatialGridContainer_findCellRange(int cellBits, float cellWidth, int nbPoints, const void* particleHash8, void* cellRange);
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
    cellBits = 16;
    nbCells = 1<<cellBits;
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
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3fTypes > >::kernel_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleIndex, void* particleHash, void* sortTmp, void* cellRange, const void* x)
{
    gpu::cuda::SpatialGridContainer3f_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);
#ifdef SOFA_DEV
    radixSort((unsigned int *)particleHash, (unsigned int *)particleIndex, (unsigned int *)sortTmp, nbPoints*8, cellBits+1);
#endif
    gpu::cuda::SpatialGridContainer_findCellRange(cellBits, cellWidth, nbPoints, particleHash, cellRange);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3f1Types > >::kernel_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleIndex, void* particleHash, void* sortTmp, void* cellRange, const void* x)
{
    gpu::cuda::SpatialGridContainer3f1_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);
#ifdef SOFA_DEV
    radixSort((unsigned int *)particleHash, (unsigned int *)particleIndex, (unsigned int *)sortTmp, nbPoints*8, cellBits+1);
#endif
    gpu::cuda::SpatialGridContainer_findCellRange(cellBits, cellWidth, nbPoints, particleHash, cellRange);
}
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
    particleIndex.recreate(nbPoints*8,8*BSIZE);
    particleHash.recreate(nbPoints*8,8*BSIZE);
#ifdef SOFA_DEV
    sortTmp.recreate(radixSortTempStorage(nbPoints*8));
#endif
    cellRange.recreate(nbCells);
    //sortedPos.recreate(nbPoints);
    kernel_updateGrid(cellBits, cellWidth, nbPoints, particleIndex.deviceWrite(), particleHash.deviceWrite(), sortTmp.deviceWrite(), cellRange.deviceWrite(), x.deviceRead());
    //kernel_reorderData(nbPoints, particleHash.deviceRead(), sortedPos.deviceWrite(), x.deviceRead());
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::draw()
{
    if (!lastX) return;
    int nbPoints = particleHash.size();
    glDisable(GL_LIGHTING);
    glColor4f(1,1,1,1);
    glPointSize(3);
    glBegin(GL_POINTS);
    for (int i=0; i<nbPoints; i++)
    {
        unsigned int cell = particleHash[i];
        unsigned int p = particleIndex[i];
        if (!(cell&1)) continue; // this is a ghost particle from a neighbor cell
        cell>>=1;
        //if (cell != 0 && cell != 65535)
        //    std::cout << i << ": "<<p<<" -> "<<cell<<", "<<(*lastX)[p]<<" -> "<<sortedPos[i]<<std::endl;
        int r = cell&3;
        int g = (cell>>2)&3;
        int b = (cell>>4)&3;
        glColor4ub(63+r*64,63+g*64,63+b*64,255);
        //glVertex3fv(sortedPos[i].ptr());
        glVertex3fv((*lastX)[p].ptr());
    }
    glEnd();
    glPointSize(1);
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
