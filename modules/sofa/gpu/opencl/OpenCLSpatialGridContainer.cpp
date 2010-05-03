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
#include <sofa/gpu/opencl/OpenCLSpatialGridContainer.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace container
{

using namespace sofa::defaulttype;
using namespace sofa::gpu::opencl;
using namespace core::componentmodel::behavior;


SOFA_DECL_CLASS(OpenCLSpatialGridContainer)

int SpatialGridContainerOpenCLClass = core::RegisterObject("GPU support using OpenCL.")
        .add< SpatialGridContainer<OpenCLVec3fTypes> >()
        ;

template class SpatialGridContainer< OpenCLVec3fTypes >;
template class SpatialGrid< SpatialGridTypes< OpenCLVec3fTypes > >;



template class SpatialGridContainer< OpenCLVec3dTypes >;
template class SpatialGrid< SpatialGridTypes< OpenCLVec3dTypes > >;



} // namespace container

} // namespace component

namespace gpu
{

namespace opencl
{

int SpatialGridContainer_RadixSortTempStorage(unsigned int numElements)
{
    NOT_IMPLEMENTED();

}

void SpatialGridContainer_RadixSort(sofa::gpu::opencl::_device_pointer /*keys*/,
        sofa::gpu::opencl::_device_pointer /*values*/,
        sofa::gpu::opencl::_device_pointer /*temp*/,
        unsigned int /*numElements*/,
        unsigned int /*keyBits*/,
        bool         /*flipBits*/)
{
    NOT_IMPLEMENTED();

}

void SpatialGridContainer3f_computeHash(int /*cellBits*/, float /*cellWidth*/, int /*nbPoints*/,gpu::opencl::_device_pointer /*particleIndex8*/,gpu::opencl::_device_pointer /*particleHash8*/, const gpu::opencl::_device_pointer /*x*/) {NOT_IMPLEMENTED();}
void SpatialGridContainer3f1_computeHash(int /*cellBits*/, float /*cellWidth*/, int /*nbPoints*/,gpu::opencl::_device_pointer /*particleIndex8*/,gpu::opencl::_device_pointer /*particleHash8*/, const gpu::opencl::_device_pointer /*x*/) {NOT_IMPLEMENTED();}
void SpatialGridContainer_findCellRange(int /*cellBits*/, int /*index0*/, float /*cellWidth*/, int /*nbPoints*/, const gpu::opencl::_device_pointer /*particleHash8*/,gpu::opencl::_device_pointer /*cellRange*/,gpu::opencl::_device_pointer /*cellGhost*/) {NOT_IMPLEMENTED();}
//void SpatialGridContainer3f_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x){NOT_IMPLEMENTED();}
//void SpatialGridContainer3f1_reorderData(int nbPoints, const gpu::opencl::_device_pointer particleHash,gpu::opencl::_device_pointer sorted, const gpu::opencl::_device_pointer x){NOT_IMPLEMENTED();}


}	//opencl
}	//gpu

} // namespace sofa
