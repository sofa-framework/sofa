/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#ifndef SOFA_GPU_CUDA_CUDASPATIALGRIDCONTAINER_H
#define SOFA_GPU_CUDA_CUDASPATIALGRIDCONTAINER_H

#include <SofaSphFluid/SpatialGridContainer.h>
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;

template<class TCoord, class TDeriv, class TReal>
class SpatialGrid< SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > >
{
public:
    typedef SpatialGridTypes < gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::CellData CellData;
    typedef typename DataTypes::GridData GridData;
    //typedef typename DataTypes::NeighborListener NeighborListener;
    typedef typename DataTypes::ParticleField ParticleField;

    enum
    {
        HASH_PX = 73856093,
        HASH_PY = 19349663,
        HASH_PZ = 83492791,
    };

public:
    SpatialGrid(Real cellWidth);

    ~SpatialGrid();

    void update(const VecCoord& x);

    void draw(const core::visual::VisualParams*);

    template<class NeighborListener>
    void findNeighbors(NeighborListener* dest, Real dist);

    void computeField(ParticleField* field, Real dist);

    /// Change particles ordering inside a given cell have contiguous indices
    ///
    /// Fill the old2new and new2old arrays giving the permutation to apply
    void reorderIndices(helper::vector<unsigned int>* old2new, helper::vector<unsigned int>* new2old);
    GridData data;

    Real getCellWidth() const { return cellWidth; }
    Real getInvCellWidth() const { return invCellWidth; }

    int getCellBits() const { return cellBits; }
    int getNbCells() const { return nbCells; }

    int getCell(const Coord& c) const
    {
        return ( (helper::rfloor(c[0]*invCellWidth*0.5f) * HASH_PX) ^
                (helper::rfloor(c[1]*invCellWidth*0.5f) * HASH_PY) ^
                (helper::rfloor(c[2]*invCellWidth*0.5f) * HASH_PZ) ) & ((1 << cellBits)-1);
    }

    //const sofa::gpu::cuda::CudaVector< unsigned int >& getParticleIndexVector() const { return particleIndex; }
    const sofa::gpu::cuda::CudaVector< int >& getCellsVector() const { return cells; }
    const sofa::gpu::cuda::CudaVector< int >& getCellGhostVector() const { return cellGhost; }

protected:
    const Real cellWidth;
    const Real invCellWidth;
    int cellBits, nbCells;
    sofa::gpu::cuda::CudaVector< unsigned int > /*particleIndex,*/ particleHash;
    //sofa::gpu::cuda::CudaVector< int > cellRange;
    sofa::gpu::cuda::CudaVector< int > cells;
    sofa::gpu::cuda::CudaVector< int > cellGhost;
    sofa::gpu::cuda::CudaVector< sofa::gpu::cuda::Vec3f1 > sortedPos;
    const VecCoord* lastX;

    void kernel_computeHash(
        int cellBits, Real cellWidth, int nbPoints, void* particleIndex, void* particleHash, const void* x);

    void kernel_updateGrid(
        int cellBits, int index0, Real cellWidth, int nbPoints, const void* particleHash,
        void* cells, void* cellGhost);
    //void kernel_reorderData(int nbPoints, const void* particleIndex, const void* particleHash, void* sorted, const void* x);

};

} // namespace container

} // namespace component

} // namespace sofa

#endif
