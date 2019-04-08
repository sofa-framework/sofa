/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "HexaRemover.inl"
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace topology
{

int HexaRemoverClass = core::RegisterObject ( "Hexahedra removing using volumetric collision detection." )
#ifdef SOFA_FLOAT
        .add< HexaRemover<defaulttype::Vec3fTypes> >(true)
#else
        .add< HexaRemover<defaulttype::Vec3dTypes> >(true)
#endif
//                                                                  .add< HexaRemover<gpu::cuda::CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
//                                                                  .add< HexaRemover<gpu::cuda::CudaVec3dTypes> >()
#endif
        ;

template class HexaRemover<defaulttype::Vec3Types> ;

//                                                                  template class HexaRemover<gpu::cuda::CudaVec3fTypes> ;
#ifdef SOFA_GPU_CUDA_DOUBLE
//                                                                  template class HexaRemover<gpu::cuda::CudaVec3dTypes> ;
#endif

}

}

}
