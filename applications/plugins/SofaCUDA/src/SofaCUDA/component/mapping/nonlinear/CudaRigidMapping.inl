/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <SofaCUDA/component/mapping/nonlinear/CudaRigidMapping.h>
#include <sofa/component/mapping/nonlinear/RigidMapping.inl>
#include <sofa/helper/accessor.h>

namespace sofa::gpu::cuda
{

using sofa::type::Mat3x3f;
using sofa::type::Vec3f;

extern "C"
{
    void RigidMappingCuda3f_apply(unsigned int size, const Mat3x3f& rotation, const Vec3f& translation, void* out, void* rotated, const void* in);
    void RigidMappingCuda3f_applyJ(unsigned int size, const Vec3f& v, const Vec3f& omega, void* out, const void* rotated);
    void RigidMappingCuda3f_applyJT(unsigned int size, unsigned int nbloc, void* out, const void* rotated, const void* in);
}

} // namespace sofa::gpu::cuda

namespace sofa::component::mapping::nonlinear
{

using namespace gpu::cuda;

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    Vec3f translation;
    Mat rotation;
    m_rotatedPoints.resize(points.size());
    out.recreate(points.size());

    translation = in[d_index.getValue()].getCenter();
    in[d_index.getValue()].writeRotationMatrix(rotation);

    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), m_rotatedPoints.deviceWrite(), points.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    gpu::cuda::CudaVec3fTypes::Deriv v, omega;
    out.recreate(points.size());
    v = getVCenter(in[d_index.getValue()]);
    omega = getVOrientation(in[d_index.getValue()]);

    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), m_rotatedPoints.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    gpu::cuda::CudaVec3fTypes::Deriv v,omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    if (nbloc > 512) nbloc = 512;
    m_data.tmp.recreate(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), nbloc, m_data.tmp.deviceWrite(), m_rotatedPoints.deviceRead(), in.deviceRead());
    const helper::ReadAccessor<gpu::cuda::CudaVec3fTypes::VecDeriv> tmp = m_data.tmp;
    for(int i=0; i<nbloc; i++)
    {
        v += tmp[2*i];
        omega += tmp[2*i+1];
    }
    getVCenter(out[d_index.getValue()]) += v;
    getVOrientation(out[d_index.getValue()]) += omega;

    dOut.endEdit();
}

//////// Rigid3d ////////
template <>
void RigidMapping<defaulttype::Rigid3Types, gpu::cuda::CudaVec3Types>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    type::Vec3 translation;
    sofa::type::Mat<3,3, defaulttype::Rigid3Types::Real> rotation;
    m_rotatedPoints.resize(points.size());
    out.recreate(points.size());

    translation = in[d_index.getValue()].getCenter();
    in[d_index.getValue()].writeRotationMatrix(rotation);

    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), m_rotatedPoints.deviceWrite(), points.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3Types, gpu::cuda::CudaVec3Types>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    gpu::cuda::CudaVec3fTypes::Deriv v, omega;
    out.recreate(points.size());
    v = getVCenter(in[d_index.getValue()]);
    omega = getVOrientation(in[d_index.getValue()]);

    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), m_rotatedPoints.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3Types, gpu::cuda::CudaVec3Types>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    gpu::cuda::CudaVec3fTypes::Deriv v, omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    if (nbloc > 512) nbloc = 512;
    m_data.tmp.recreate(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), nbloc, m_data.tmp.deviceWrite(), m_rotatedPoints.deviceRead(), in.deviceRead());
    const helper::ReadAccessor<gpu::cuda::CudaVec3Types::VecDeriv> tmp = m_data.tmp;
    for(int i=0; i<nbloc; i++)
    {
        v += tmp[2*i];
        omega += tmp[2*i+1];
    }
    getVCenter(out[d_index.getValue()]) += v;
    getVOrientation(out[d_index.getValue()]) += omega;

    dOut.endEdit();
}

//////// Rigid3f ////////

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    Vec3f translation;
    Mat rotation;
    m_rotatedPoints.resize(points.size());
    out.recreate(points.size());

    translation = in[d_index.getValue()].getCenter();
    in[d_index.getValue()].writeRotationMatrix(rotation);

    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), m_rotatedPoints.deviceWrite(), points.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    gpu::cuda::CudaVec3fTypes::Deriv v, omega;
    out.recreate(points.size());
    v = getVCenter(in[d_index.getValue()]);
    omega = getVOrientation(in[d_index.getValue()]);

    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), m_rotatedPoints.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();

    const auto& points = this->d_points.getValue();
    gpu::cuda::CudaVec3fTypes::Deriv v, omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    if (nbloc > 512) nbloc = 512;
    m_data.tmp.recreate(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), nbloc, m_data.tmp.deviceWrite(), m_rotatedPoints.deviceRead(), in.deviceRead());
    const helper::ReadAccessor<gpu::cuda::CudaVec3fTypes::VecDeriv> tmp = m_data.tmp;
    for(int i=0; i<nbloc; i++)
    {
        v += tmp[2*i];
        omega += tmp[2*i+1];
    }
    getVCenter(out[d_index.getValue()]) += v;
    getVOrientation(out[d_index.getValue()]) += omega;

    dOut.endEdit();
}

} // namespace sofa::component::mapping::nonlinear
