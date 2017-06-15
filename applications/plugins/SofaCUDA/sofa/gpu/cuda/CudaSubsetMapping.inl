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
#ifndef SOFA_GPU_CUDA_CUDASUBSETMAPPING_INL
#define SOFA_GPU_CUDA_CUDASUBSETMAPPING_INL

#include "CudaSubsetMapping.h"
#include <SofaBaseMechanics/SubsetMapping.inl>

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
    const IndexArray& map = this->f_indices.getValue();
    if (!map.empty())
    {
        this->data.init(this->fromModel->getSize(), map);
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f_apply(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f_applyJ(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    const IndexArray& map = this->f_indices.getValue();
    if (map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f_applyJT1(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

//////// CudaVec3f1

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::postInit()
{
    const IndexArray& map = this->f_indices.getValue();
    if (!map.empty())
    {
        this->data.init(this->fromModel->getSize(), map);
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f1_apply(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f1_applyJ(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    const IndexArray& map = this->f_indices.getValue();
    if (map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f1_applyJT1(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f1_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::postInit()
{
    const IndexArray& map = this->f_indices.getValue();
    if (!map.empty())
    {
        this->data.init(this->fromModel->getSize(), map);
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f1_3f_apply(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f1_3f_applyJ(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    const IndexArray& map = this->f_indices.getValue();
    if (map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f1_3f_applyJT1(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f1_3f_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::postInit()
{
    const IndexArray& map = this->f_indices.getValue();
    if (!map.empty())
    {
        this->data.init(this->fromModel->getSize(), map);
    }
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f_3f1_apply(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    const IndexArray& map = this->f_indices.getValue();
    out.fastResize(map.size());
    SubsetMappingCuda3f_3f1_applyJ(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    const IndexArray& map = this->f_indices.getValue();
    if (map.size() == 0) return;

    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    unsigned int insize = out.size();
    if (data.mapT.empty())
        SubsetMappingCuda3f_3f1_applyJT1(map.size(), map.deviceRead(), out.deviceWrite(), in.deviceRead());
    else
        SubsetMappingCuda3f_3f1_applyJT(insize, data.maxNOut, data.mapT.deviceRead(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
