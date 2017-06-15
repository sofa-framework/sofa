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
#include "HexaRemover.inl"
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS ( HexaRemover );

int HexaRemoverClass = core::RegisterObject ( "Hexahedra removing using volumetric collision detection." )
#ifdef SOFA_FLOAT
        .add< HexaRemover<defaulttype::Vec3fTypes> >(true)
#else
        .add< HexaRemover<defaulttype::Vec3dTypes> >(true)
#ifndef SOFA_DOUBLE
        .add< HexaRemover<defaulttype::Vec3fTypes> >()
#endif
#endif
//                                                                  .add< HexaRemover<gpu::cuda::CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
//                                                                  .add< HexaRemover<gpu::cuda::CudaVec3dTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class HexaRemover<defaulttype::Vec3dTypes> ;
#endif
#ifndef SOFA_DOUBLE
template class HexaRemover<defaulttype::Vec3fTypes> ;
#endif
//                                                                  template class HexaRemover<gpu::cuda::CudaVec3fTypes> ;
#ifdef SOFA_GPU_CUDA_DOUBLE
//                                                                  template class HexaRemover<gpu::cuda::CudaVec3dTypes> ;
#endif

}

}

}
