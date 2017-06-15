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

#ifndef SOFAOPENCL_OPENCLSPATIALGRIDCONTAINER_H
#define SOFAOPENCL_OPENCLSPATIALGRIDCONTAINER_H

#include <SofaSphFluid/SpatialGridContainer.h>
#include "OpenCLTypes.h"
#include <sofa/defaulttype/Vec.h>
#include "oclRadixSort/RadixSort.h"

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;

template<class TCoord, class TDeriv, class TReal>
class SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >
{
public:
    typedef SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > DataTypes;
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

    RadixSort *radixsort;

public:
    SpatialGrid(Real cellWidth);

    void update(const VecCoord& x);

    void draw(const sofa::core::visual::VisualParams*);

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

    //const sofa::gpu::opencl::OpenCLVector< unsigned int >& getParticleIndexVector() const { return particleIndex; }
    const sofa::gpu::opencl::OpenCLVector< int >& getCellsVector() const { return cells; }
    const sofa::gpu::opencl::OpenCLVector< int >& getCellGhostVector() const { return cellGhost; }

protected:
    const Real cellWidth;
    const Real invCellWidth;
    int cellBits, nbCells;
    sofa::gpu::opencl::OpenCLVector< unsigned int > /*particleIndex,*/ particleHash, sortTmp;
    //sofa::gpu::opencl::OpenCLVector< int > cellRange;
    sofa::gpu::opencl::OpenCLVector< int > cells;
    sofa::gpu::opencl::OpenCLVector< int > cellGhost;
    sofa::gpu::opencl::OpenCLVector< sofa::gpu::opencl::Vec3f1 > sortedPos;
    const VecCoord* lastX;

    static void kernel_updateGrid(int cellBits, int index0, Real cellWidth, int nbPoints, gpu::opencl::_device_pointer particleIndex, gpu::opencl::_device_pointer particleHash, gpu::opencl::_device_pointer sortTmp, gpu::opencl::_device_pointer cells, gpu::opencl::_device_pointer cellGhost, gpu::opencl::_device_pointer x,RadixSort *rs);
    //static void kernel_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleIndex, const gpu::opencl::_device_pointer particleHash, gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x);

};

} // namespace container

} // namespace component

} // namespace sofa

#endif
