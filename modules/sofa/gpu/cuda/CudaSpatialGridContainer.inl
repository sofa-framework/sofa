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

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SpatialGridContainer3f_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x);
    void SpatialGridContainer3f1_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x);
    void SpatialGridContainer3f_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x);
    void SpatialGridContainer3f1_reorderData(int nbPoints, const void* particleHash, void* sorted, const void* x);
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
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3fTypes > >::kernel_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x)
{
    gpu::cuda::SpatialGridContainer3f_updateGrid(cellBits, cellWidth, nbPoints, particleHash, sortTmp, cellStart, x);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVec3f1Types > >::kernel_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, const void* x)
{
    gpu::cuda::SpatialGridContainer3f1_updateGrid(cellBits, cellWidth, nbPoints, particleHash, sortTmp, cellStart, x);
}

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

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::update(const VecCoord& x)
{
    lastX = &x;
    data.clear();
    int nbPoints = x.size();
    particleHash.fastResize(nbPoints);
    sortTmp.fastResize(nbPoints);
    cellStart.fastResize(nbCells);
    sortedPos.fastResize(nbPoints);
    kernel_updateGrid(cellBits, cellWidth, nbPoints, particleHash.deviceWrite(), sortTmp.deviceWrite(), cellStart.deviceWrite(), x.deviceRead());
    kernel_reorderData(nbPoints, particleHash.deviceRead(), sortedPos.deviceWrite(), x.deviceRead());
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
        unsigned int cell = particleHash[i][0];
        //unsigned int p = particleHash[i][1];
        //if (cell != 0 && cell != 65535)
        //    std::cout << i << ": "<<p<<" -> "<<cell<<", "<<(*lastX)[p]<<" -> "<<sortedPos[i]<<std::endl;
        int r = cell&3;
        int g = (cell>>2)&3;
        int b = (cell>>4)&3;
        glColor4ub(63+r*64,63+g*64,63+b*64,255);
        glVertex3fv(sortedPos[i].ptr());
        //glVertex3fv((*lastX)[p].ptr());
    }
    glEnd();
    glPointSize(1);
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
