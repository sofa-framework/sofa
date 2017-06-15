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
#ifndef SOFAOPENCL_OPENCLSPHFLUIDFORCEFIELD_H
#define SOFAOPENCL_OPENCLSPHFLUIDFORCEFIELD_H

#include "OpenCLTypes.h"
#include <SofaSphFluid/SPHFluidForceField.h>
#include "OpenCLSpatialGridContainer.h"

namespace sofa
{

namespace gpu
{

namespace opencl
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

} // namespace opencl

} // namespace gpu

namespace component
{

namespace forcefield
{

template <class TCoord, class TDeriv, class TReal>
class SPHFluidForceFieldInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef SPHFluidForceFieldInternalData<DataTypes> Data;
    typedef SPHFluidForceField<DataTypes> Main;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    gpu::opencl::GPUSPHFluid<Real> params;
    gpu::opencl::OpenCLVector<defaulttype::Vec4f> pos4;

    void fillParams(Main* m, int kernelType, double kFactor=1.0, double bFactor=1.0)
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

        if (kernelType == 1)
        {
            params.CWd          = SPHKernel<SPH_KERNEL_CUBIC,Coord>::constW(h);
            //params.CgradWd      = SPHKernel<SPH_KERNEL_CUBIC,Coord>::constGradW(h);
            params.CgradWp      = SPHKernel<SPH_KERNEL_CUBIC,Coord>::constGradW(h);
            params.ClaplacianWv = SPHKernel<SPH_KERNEL_CUBIC,Coord>::constLaplacianW(h);
        }
        else
        {
            params.CWd          = SPHKernel<SPH_KERNEL_DEFAULT_DENSITY,Coord>::constW(h);
            //params.CgradWd      = SPHKernel<SPH_KERNEL_DEFAULT_DENSITY,Coord>::constGradW(h);
            params.CgradWp      = SPHKernel<SPH_KERNEL_DEFAULT_PRESSURE,Coord>::constGradW(h);
            params.ClaplacianWv = SPHKernel<SPH_KERNEL_DEFAULT_VISCOSITY,Coord>::constLaplacianW(h);
        }
    }

    void Kernels_computeDensity(int gsize, const gpu::opencl::_device_pointer cells, const gpu::opencl::_device_pointer cellGhost, gpu::opencl::_device_pointer pos4, const gpu::opencl::_device_pointer x);
    void Kernels_addForce(int gsize, const gpu::opencl::_device_pointer cells, const gpu::opencl::_device_pointer cellGhost, gpu::opencl::_device_pointer f, const gpu::opencl::_device_pointer pos4, const gpu::opencl::_device_pointer vel);
    void Kernels_addDForce(int gsize, const gpu::opencl::_device_pointer cells, const gpu::opencl::_device_pointer cellGhost, gpu::opencl::_device_pointer f, const gpu::opencl::_device_pointer pos4, const gpu::opencl::_device_pointer dx, const gpu::opencl::_device_pointer vel);
};

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3fTypes>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3fTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3fTypes>::draw(const sofa::core::visual::VisualParams* vparams);

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3dTypes>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3dTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
