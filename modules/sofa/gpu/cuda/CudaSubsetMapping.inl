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
#ifndef SOFA_GPU_CUDA_CUDASUBSETMAPPING_INL
#define SOFA_GPU_CUDA_CUDASUBSETMAPPING_INL

#include "CudaSubsetMapping.h"
#include <sofa/component/mapping/SubsetMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SubsetMappingCuda3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f1_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f1_3f_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f1_3f_applyJT1(unsigned int size, const void* map, void* out, const void* in);

    void SubsetMappingCuda3f_3f1_apply(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJ(unsigned int size, const void* map, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJT(unsigned int insize, unsigned int maxNOut, const void* mapT, void* out, const void* in);
    void SubsetMappingCuda3f_3f1_applyJT1(unsigned int size, const void* map, void* out, const void* in);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::postInit()
{
    const IndexArray& indices = this->f_indices.getValue();
    if (!indices.empty())
    {
        this->data.clear(indices.size());
        for (unsigned int i=0; i<indices.size(); i++)
            this->data.addPoint(indices[i]);
        this->data.init(this->fromModel->getX()->size());
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_applyJ(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    if (data.map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f_applyJT1(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

//////// CudaVec3f1

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::postInit()
{
    const IndexArray& indices = this->f_indices.getValue();
    if (!indices.empty())
    {
        this->data.clear(indices.size());
        for (unsigned int i=0; i<indices.size(); i++)
            this->data.addPoint(indices[i]);
        this->data.init(this->fromModel->getX()->size());
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f1_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f1_applyJ(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    if (data.map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f1_applyJT1(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f1_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::postInit()
{
    const IndexArray& indices = this->f_indices.getValue();
    if (!indices.empty())
    {
        this->data.clear(indices.size());
        for (unsigned int i=0; i<indices.size(); i++)
            this->data.addPoint(indices[i]);
        this->data.init(this->fromModel->getX()->size());
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f1_3f_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f1_3f_applyJ(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    if (data.map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f1_3f_applyJT1(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f1_3f_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::postInit()
{
    const IndexArray& indices = this->f_indices.getValue();
    if (!indices.empty())
    {
        this->data.clear(indices.size());
        for (unsigned int i=0; i<indices.size(); i++)
            this->data.addPoint(indices[i]);
        this->data.init(this->fromModel->getX()->size());
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_3f1_apply(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(data.map.size());
    SubsetMappingCuda3f_3f1_applyJ(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    if (data.map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f_3f1_applyJT1(data.map.size(), data.map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f_3f1_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
