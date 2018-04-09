/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GPU_CUDA_CUDAEXTERNALFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAEXTERNALFORCEFIELD_INL

#include "CudaExternalForceField.h"
#include <sofa/component/interactionforcefield/ExternalForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void ExternalForceFieldCuda3f_addForce(unsigned int size,void* f, const void* indices,const void *forces );

};

} // namespace cuda

} // namespace gpu



namespace component
{

namespace interactionforcefield
{

using namespace gpu::cuda;

template<>
void ExternalForceField<sofa::gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord&/* p*/, const VecDeriv& /*v*/)
{
    gpu::cuda::CudaVector<unsigned> indices;
    unsigned n=m_indices.getValue().size();
    indices.resize(n);
    for(unsigned i=0; i<m_indices.getValue().size(); i++)
        indices[i]=m_indices.getValue()[i];
    sofa::gpu::cuda::ExternalForceFieldCuda3f_addForce(m_indices.getValue().size(), f.deviceWrite(),indices.deviceRead(),m_forces.getValue().deviceRead());
}


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
