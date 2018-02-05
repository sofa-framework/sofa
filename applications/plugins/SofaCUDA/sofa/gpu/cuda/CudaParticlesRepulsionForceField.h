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
#ifndef SOFA_GPU_CUDA_CUDAPARTICLESREPULSIONFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAPARTICLESREPULSIONFORCEFIELD_H

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaSphFluid/ParticlesRepulsionForceField.h>
#include <sofa/gpu/cuda/CudaSpatialGridContainer.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class real>
struct GPURepulsion
{
    real d;
    real d2;
    real stiffness;
    real damping;
};

typedef GPURepulsion<float> GPURepulsion3f;
typedef GPURepulsion<double> GPURepulsion3d;

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
