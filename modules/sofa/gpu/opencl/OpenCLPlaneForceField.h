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
#ifndef SOFA_GPU_OPENCL_OPENCLPLANEFORCEFIELD_H
#define SOFA_GPU_OPENCL_OPENCLPLANEFORCEFIELD_H

#include "OpenCLTypes.h"
#include <sofa/component/forcefield/PlaneForceField.h>

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
void PlaneForceField<gpu::opencl::OpenCLVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3f1Types>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3f1Types>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);



template <>
void PlaneForceField<gpu::opencl::OpenCLVec3dTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3dTypes>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3d1Types>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void PlaneForceField<gpu::opencl::OpenCLVec3d1Types>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

}
}

} // namespace sofa

#endif
