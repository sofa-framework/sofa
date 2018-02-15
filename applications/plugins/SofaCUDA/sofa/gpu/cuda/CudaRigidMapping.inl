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
#ifndef SOFA_GPU_CUDA_CUDARIGIDMAPPING_INL
#define SOFA_GPU_CUDA_CUDARIGIDMAPPING_INL

#include "CudaRigidMapping.h"
#include <SofaRigid/RigidMapping.inl>
#include <sofa/helper/accessor.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

using sofa::defaulttype::Mat3x3f;
using sofa::defaulttype::Vec3f;

extern "C"
{
    void RigidMappingCuda3f_apply(unsigned int size, const Mat3x3f& rotation, const Vec3f& translation, void* out, void* rotated, const void* in);
    void RigidMappingCuda3f_applyJ(unsigned int size, const Vec3f& v, const Vec3f& omega, void* out, const void* rotated);
    void RigidMappingCuda3f_applyJT(unsigned int size, unsigned int nbloc, void* out, const void* rotated, const void* in);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Coord translation;
    Mat rotation;
    rotatedPoints.resize(points.size());
    out.recreate(points.size());

    translation = in[index.getValue()].getCenter();
    in[index.getValue()].writeRotationMatrix(rotation);

    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    rotatedPoints[i] = rotation*points[i];
    //    out[i] = rotatedPoints[i];
    //    out[i] += translation;
    //}
    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), rotatedPoints.deviceWrite(), points.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Deriv v,omega;
    out.recreate(points.size());
    v = getVCenter(in[index.getValue()]);
    omega = getVOrientation(in[index.getValue()]);
    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    // out = J in
    //    // J = [ I -OM^ ]
    //    out[i] =  v - cross(rotatedPoints[i],omega);
    //}
    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), rotatedPoints.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Deriv v,omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    if (nbloc > 512) nbloc = 512;
    data.tmp.recreate(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), nbloc, data.tmp.deviceWrite(), rotatedPoints.deviceRead(), in.deviceRead());
    helper::ReadAccessor<gpu::cuda::CudaVec3fTypes::VecDeriv> tmp = data.tmp;
    for(int i=0; i<nbloc; i++)
    {
        v += tmp[2*i];
        omega += tmp[2*i+1];
    }
    getVCenter(out[index.getValue()]) += v;
    getVOrientation(out[index.getValue()]) += omega;

    dOut.endEdit();
}

//////// Rigid3d ////////
#ifndef SOFA_FLOAT
template <>
void RigidMapping<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Coord translation;
    Mat rotation;
    rotatedPoints.resize(points.size());
    out.recreate(points.size());

    translation = in[index.getValue()].getCenter();
    in[index.getValue()].writeRotationMatrix(rotation);

    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    rotatedPoints[i] = rotation*points[i];
    //    out[i] = rotatedPoints[i];
    //    out[i] += translation;
    //}
    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), rotatedPoints.deviceWrite(), points.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Deriv v,omega;
    out.recreate(points.size());
    v = getVCenter(in[index.getValue()]);
    omega = getVOrientation(in[index.getValue()]);
    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    // out = J in
    //    // J = [ I -OM^ ]
    //    out[i] =  v - cross(rotatedPoints[i],omega);
    //}
    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), rotatedPoints.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Deriv v,omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    if (nbloc > 512) nbloc = 512;
    data.tmp.recreate(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), nbloc, data.tmp.deviceWrite(), rotatedPoints.deviceRead(), in.deviceRead());
    helper::ReadAccessor<gpu::cuda::CudaVec3fTypes::VecDeriv> tmp = data.tmp;
    for(int i=0; i<nbloc; i++)
    {
        v += tmp[2*i];
        omega += tmp[2*i+1];
    }
    getVCenter(out[index.getValue()]) += v;
    getVOrientation(out[index.getValue()]) += omega;

    dOut.endEdit();
}
#endif
//////// Rigid3f ////////

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Coord translation;
    Mat rotation;
    rotatedPoints.resize(points.size());
    out.recreate(points.size());

    translation = in[index.getValue()].getCenter();
    in[index.getValue()].writeRotationMatrix(rotation);

    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    rotatedPoints[i] = rotation*points[i];
    //    out[i] = rotatedPoints[i];
    //    out[i] += translation;
    //}
    RigidMappingCuda3f_apply(points.size(), rotation, translation, out.deviceWrite(), rotatedPoints.deviceWrite(), points.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Deriv v,omega;
    out.recreate(points.size());
    v = getVCenter(in[index.getValue()]);
    omega = getVOrientation(in[index.getValue()]);
    //for(unsigned int i=0;i<points.size();i++)
    //{
    //    // out = J in
    //    // J = [ I -OM^ ]
    //    out[i] =  v - cross(rotatedPoints[i],omega);
    //}
    RigidMappingCuda3f_applyJ(points.size(), v, omega, out.deviceWrite(), rotatedPoints.deviceRead());

    dOut.endEdit();
}

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();

    const VecCoord& points = this->points.getValue();
    Deriv v,omega;
    int nbloc = ((points.size()+BSIZE-1)/BSIZE);
    if (nbloc > 512) nbloc = 512;
    data.tmp.recreate(2*nbloc);
    RigidMappingCuda3f_applyJT(points.size(), nbloc, data.tmp.deviceWrite(), rotatedPoints.deviceRead(), in.deviceRead());
    helper::ReadAccessor<gpu::cuda::CudaVec3fTypes::VecDeriv> tmp = data.tmp;
    for(int i=0; i<nbloc; i++)
    {
        v += tmp[2*i];
        omega += tmp[2*i+1];
    }
    getVCenter(out[index.getValue()]) += v;
    getVOrientation(out[index.getValue()]) += omega;

    dOut.endEdit();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
