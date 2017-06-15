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

#ifndef SOFAOPENCL_OPENCLSPATIALGRIDCONTAINER_INL
#define SOFAOPENCL_OPENCLSPATIALGRIDCONTAINER_INL

#include "OpenCLSpatialGridContainer.h"
#include <SofaSphFluid/SpatialGridContainer.inl>
#include <sofa/helper/gl/template.h>




namespace sofa
{

namespace gpu
{

namespace opencl
{

extern int SpatialGridContainer_RadixSortTempStorage(unsigned int numElements);

extern void SpatialGridContainer_RadixSort(sofa::gpu::opencl::_device_pointer keys,
        sofa::gpu::opencl::_device_pointer values,
        sofa::gpu::opencl::_device_pointer temp,
        unsigned int numElements,
        unsigned int keyBits = 32,
        bool         flipBits = false);

extern void SpatialGridContainer3f_computeHash(int cellBits, float cellWidth, int nbPoints,gpu::opencl::_device_pointer particleIndex8,gpu::opencl::_device_pointer particleHash8, const gpu::opencl::_device_pointer x);
extern void SpatialGridContainer3f1_computeHash(int cellBits, float cellWidth, int nbPoints,gpu::opencl::_device_pointer particleIndex8,gpu::opencl::_device_pointer particleHash8, const gpu::opencl::_device_pointer x);
extern void SpatialGridContainer_findCellRange(int /*cellBits*/, int /*index0*/, float /*cellWidth*/, int /*nbPoints*/, const gpu::opencl::_device_pointer /*particleHash8*/,gpu::opencl::_device_pointer /*cellRange*/,gpu::opencl::_device_pointer /*cellGhost*/);

//extern void SpatialGridContainer3f_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x);
//extern void SpatialGridContainer3f1_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x);


} // namespace opencl

} // namespace gpu

namespace component
{

namespace container
{

using namespace sofa::helper;

//      template<class TCoord, class TDeriv, class TReal>
//      typename SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::Grid SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::emptyGrid;

template<class TCoord, class TDeriv, class TReal>
SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::SpatialGrid(Real cellWidth)
    : cellWidth(cellWidth), invCellWidth(1/cellWidth), lastX(NULL)
{
    cellBits = 15;
    nbCells = 1<<cellBits;
    radixsort = NULL;
}

template<class TCoord, class TDeriv, class TReal> template<class NeighborListener>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::findNeighbors(NeighborListener* /*dest*/, Real /*dist*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::findNeighbors(NeighborListener* dest, Real dist)"<<std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::computeField(ParticleField* /*field*/, Real /*dist*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::computeField(ParticleField* field, Real dist)"<<std::endl;
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::reorderIndices(helper::vector<unsigned int>* /*old2new*/, helper::vector<unsigned int>* /*new2old*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::reorderIndices(helper::vector<unsigned int>* old2new, helper::vector<unsigned int>* new2old)"<<std::endl;
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVec3fTypes > >::kernel_updateGrid(int cellBits, int index0, Real cellWidth, int nbPoints,gpu::opencl::_device_pointer particleIndex,gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer /*sortTmp*/,gpu::opencl::_device_pointer cells,gpu::opencl::_device_pointer cellGhost, gpu::opencl::_device_pointer x,RadixSort *rs)
{
    gpu::opencl::SpatialGridContainer3f_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);

    int nbbits = 8;
    while (nbbits < cellBits + 1) ++nbbits;
//	printf("nbbits: %d\n",nbbits);

//	printf("nbpoint1%d\n",nbPoints);
//	gpu::opencl::SpatialGridContainer_RadixSort(particleHash, particleIndex, sortTmp, nbPoints*8, nbbits);
    rs->sort(particleHash,particleIndex,nbPoints*8,nbbits+1);

    gpu::opencl::SpatialGridContainer_findCellRange(cellBits, index0, cellWidth, nbPoints, particleHash, cells, cellGhost);


}

template<>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVec3f1Types > >::kernel_updateGrid(int cellBits, int index0, Real cellWidth, int nbPoints,gpu::opencl::_device_pointer particleIndex,gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer /*sortTmp*/,gpu::opencl::_device_pointer cells,gpu::opencl::_device_pointer cellGhost, gpu::opencl::_device_pointer x,RadixSort *rs)
{
    gpu::opencl::SpatialGridContainer3f1_computeHash(cellBits, cellWidth, nbPoints, particleIndex, particleHash, x);

    int nbbits = 8;
    while (nbbits < cellBits + 1) ++nbbits;
//	gpu::opencl::SpatialGridContainer_RadixSort(particleHash, particleIndex, sortTmp, nbPoints*8, nbbits);
//	printf("nbpoint2%d\n",nbPoints);
    rs->sort(particleIndex,particleHash,nbPoints*8,nbbits);
//	gpu::opencl::SpatialGridContainer_RadixSort(particleHash, particleIndex, sortTmp, nbPoints*8, nbbits);

    gpu::opencl::SpatialGridContainer_findCellRange(cellBits, index0, cellWidth, nbPoints, particleHash, cells, cellGhost);
}



template<>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVec3dTypes > >::kernel_updateGrid(int /*cellBits*/, int /*index0*/, Real /*cellWidth*/, int /*nbPoints*/,gpu::opencl::_device_pointer /*particleIndex*/,gpu::opencl::_device_pointer /*particleHash*/,gpu::opencl::_device_pointer /*sortTmp*/,gpu::opencl::_device_pointer /*cells*/,gpu::opencl::_device_pointer /*cellGhost*/, const gpu::opencl::_device_pointer /*x*/,RadixSort */*rs*/)
{
    std::cerr << "TODO: SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVec3dTypes > >::kernel_updateGrid(int cellBits, int index0, Real cellWidth, int nbPoints,gpu::opencl::_device_pointer particleIndex,gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sortTmp,gpu::opencl::_device_pointer cells,gpu::opencl::_device_pointer cellGhost, const gpu::opencl::_device_pointer x)"<<std::endl;
}

/*
template<>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVec3fTypes > >::kernel_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x)
{
	gpu::opencl::SpatialGridContainer3f_reorderData(nbPoints, particleHash, sorted, x);
}

template<>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVec3f1Types > >::kernel_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x)
{
	gpu::opencl::SpatialGridContainer3f1_reorderData(nbPoints, particleHash, sorted, x);
}
*/

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::update(const VecCoord& x)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<void*>::BSIZE;
    lastX = &x;
    data.clear();
    int nbPoints = x.size();
    int index0 = nbCells+ BSIZE;
    /*particleIndex*/ cells.recreate(index0+nbPoints*8,8*BSIZE);
    particleHash.recreate(nbPoints*8,8*BSIZE);
//	printf("nbpoint3%d\n",nbPoints);
    sortTmp.recreate(gpu::opencl::SpatialGridContainer_RadixSortTempStorage(nbPoints*8));
//	if(radixsort!=NULL){delete(radixsort);};
//	radixsort = new RadixSort(nbPoints*8, BSIZE, true);
    if(radixsort==NULL) {radixsort = new RadixSort(nbPoints*8, BSIZE, true);};


    //cells.recreate(nbCells+1);
    cellGhost.recreate(nbCells);
    //sortedPos.recreate(nbPoints);

    kernel_updateGrid(cellBits, index0, cellWidth*2, nbPoints, cells.deviceWriteAt(index0), particleHash.deviceWrite(), sortTmp.deviceWrite(), cells.deviceWrite(), cellGhost.deviceWrite(), x.deviceRead(),radixsort);

    /*
    std::cout << nbPoints*8 << " entries in " << nbCells << " cells." << std::endl;
    int nfill = 0;
    for (int c=0;c<nbCells;++c)
    {
    	if (cells[c] <= 0) continue;
    	std::cout << "Cell " << c << ": range = " << cells[c]-index0 << " - " << cells[c+1]-index0 << "     ghost = " << cellGhost[c] << std::endl;
    	++nfill;
    }
    std::cout << ((1000*nfill)/nbCells) * 0.1 << " % cells with particles." << std::endl;
    */
    //kernel_reorderData(nbPoints, particleHash.deviceRead(), sortedPos.deviceWrite(), x.deviceRead());
}

template<class TCoord, class TDeriv, class TReal>
void SpatialGrid< SpatialGridTypes < gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> > >::draw(const sofa::core::visual::VisualParams* /*vparams*/)
{
    int BSIZE = gpu::opencl::OpenCLMemoryManager<void*>::BSIZE;
    if (!lastX) return;
    int nbPoints = particleHash.size();
    int index0 = nbCells+BSIZE;
    glDisable(GL_LIGHTING);
    glColor4f(1,1,1,1);
    glPointSize(3);
    glBegin(GL_POINTS);
    for (int i=0; i<nbPoints; i++)
    {
        unsigned int cell = particleHash[i];
        unsigned int p = cells[index0+i]; //particleIndex[i];
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
