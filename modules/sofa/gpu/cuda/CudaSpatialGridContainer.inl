/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
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
    void SpatialGridContainer3f_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, void* x);
    void SpatialGridContainer3f1_updateGrid(int cellBits, float cellWidth, int nbPoints, void* particleHash, void* sortTmp, void* cellStart, void* x);
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
    : cellWidth(cellWidth), invCellWidth(1/cellWidth)
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

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::update(const VecCoord& x)
{
    data.clear();
    int nbPoints = x.size();
    particleHash.fastResize(nbPoints);
    sortTmp.fastResize(nbPoints);
    cellStart.resize(nbCells);

}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >::draw()
{
}

} // namespace container

} // namespace component

} // namespace sofa

#endif
