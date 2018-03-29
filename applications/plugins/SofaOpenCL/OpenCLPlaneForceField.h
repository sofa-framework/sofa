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
#ifndef SOFAOPENCL_OPENCLPLANEFORCEFIELD_H
#define SOFAOPENCL_OPENCLPLANEFORCEFIELD_H

#include "OpenCLTypes.h"
#include <SofaBoundaryCondition/PlaneForceField.h>

namespace sofa
{

namespace gpu
{

namespace opencl
{

template<class real>
struct GPUPlane
{
    defaulttype::Vec<3,real> normal;
    real d;
    real stiffness;
    real damping;
};

} // namespace opencl

} // namespace gpu

namespace component
{

namespace forcefield
{



template<class TCoord, class TDeriv, class TReal>
class PlaneForceFieldInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    typedef TReal Real;

    gpu::opencl::GPUPlane<Real> plane;
    gpu::opencl::OpenCLVector<Real> penetration;

    PlaneForceFieldInternalData()
    {}

};

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3fTypes>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3fTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3f1Types>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3f1Types>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);



template <>
void PlaneForceField<gpu::opencl::OpenCLVec3dTypes>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3dTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3d1Types>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3d1Types>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

}
}

} // namespace sofa

#endif
