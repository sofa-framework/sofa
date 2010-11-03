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
#ifndef SOFA_GPU_CUDA_CUDASPHFLUIDFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDASPHFLUIDFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/SPHFluidForceField.h>
#include <sofa/gpu/cuda/CudaSpatialGridContainer.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class real>
struct GPUSPHFluid
{
    real h;         ///< particles radius
    real h2;        ///< particles radius squared
    real stiffness; ///< pressure stiffness
    real mass;      ///< particles mass
    real mass2;     ///< particles mass squared
    real density0;  ///< 1000 kg/m3 for water
    real viscosity;
    real surfaceTension;

    // Precomputed constants for smoothing kernels
    real CWd;          ///< =     constWd(h)
    real CgradWd;      ///< = constGradWd(h)
    real CgradWp;      ///< = constGradWp(h)
    real ClaplacianWv; ///< =  constLaplacianWv(h)
    real CgradWc;      ///< = constGradWc(h)
    real ClaplacianWc; ///< =  constLaplacianWc(h)
};

typedef GPUSPHFluid<float> GPUSPHFluid3f;
typedef GPUSPHFluid<double> GPUSPHFluid3d;

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template <class TCoord, class TDeriv, class TReal>
class SPHFluidForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef SPHFluidForceFieldInternalData<DataTypes> Data;
    typedef SPHFluidForceField<DataTypes> Main;
    typedef typename DataTypes::Real Real;
    gpu::cuda::GPUSPHFluid<Real> params;
    gpu::cuda::CudaVector<defaulttype::Vec4f> pos4;

    void fillParams(Main* m, double kFactor=1.0, double bFactor=1.0)
    {
        Real h = m->particleRadius.getValue();
        params.h = h;
        params.h2 = h*h;
        params.stiffness = (Real)(kFactor*m->pressureStiffness.getValue());
        params.mass = m->particleMass.getValue();
        params.mass2 = params.mass*params.mass;
        params.density0 = m->density0.getValue();
        params.viscosity = (Real)(bFactor*m->viscosity.getValue());
        params.surfaceTension = (Real)(kFactor*m->surfaceTension.getValue());

        params.CWd          = m->constWd(h);
        params.CgradWd      = m->constGradWd(h);
        params.CgradWp      = m->constGradWp(h);
        params.ClaplacianWv = m->constLaplacianWv(h);
        params.CgradWc      = m->constGradWc(h);
        params.ClaplacianWc = m->constLaplacianWc(h);
    }

    void Kernels_computeDensity(int gsize, const void* cells, const void* cellGhost, void* pos4, const void* x);
    void Kernels_addForce(int gsize, const void* cells, const void* cellGhost, void* f, const void* pos4, const void* vel);
    void Kernels_addDForce(int gsize, const void* cells, const void* cellGhost, void* f, const void* pos4, const void* dx, const void* vel);
};


template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::addForce(DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v, const core::MechanicalParams* mparams);

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::addDForce(DataVecDeriv& d_df, const DataVecDeriv& d_dx, const core::MechanicalParams* mparams);

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::draw();

#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3dTypes>::addForce(DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v, const core::MechanicalParams* mparams);

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3dTypes>::addDForce(DataVecDeriv& d_df, const DataVecDeriv& d_dx, const core::MechanicalParams* mparams);

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
