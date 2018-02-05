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
#ifndef SOFA_GPU_CUDA_CUDASUBSETMAPPING_H
#define SOFA_GPU_CUDA_CUDASUBSETMAPPING_H

#include "CudaTypes.h"
#include <SofaBaseMechanics/SubsetMapping.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
class SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    int maxNOut;
    sofa::gpu::cuda::CudaVector<int> mapT;
    SubsetMappingInternalData() : maxNOut(0)
    {}

    template<class VecIndex>
    void init(int insize, const VecIndex& map)
    {
        unsigned int n = map.size();
        std::vector<int> nout;
        if (n==0) return;
        if (n==1)
            maxNOut = 1;
        else
        {
            // compute mapT
            nout.resize(insize);
            for (unsigned int i=0; i<map.size(); i++)
                nout[map[i]]++;
            for (int i=0; i<insize; i++)
                if (nout[i] > maxNOut) maxNOut = nout[i];
        }
        if (maxNOut <= 1)
        {
            std::cout << "CudaSubsetMapping: strict subset, no need for mapT."<<std::endl;
            // at most one duplicated points per input. mapT is not necessary
            mapT.clear();
        }
        else
        {
            int nbloc = (insize+BSIZE-1)/BSIZE;
            std::cout << "CudaSubsetMapping: mapT with "<<maxNOut<<" entries per DOF and "<<nbloc<<" blocs."<<std::endl;
            mapT.resize(nbloc*(BSIZE*maxNOut));
            for (unsigned int i=0; i<mapT.size(); i++)
                mapT[i] = -1;
            nout.clear();
            nout.resize(insize);
            for (unsigned int i=0; i<map.size(); i++)
            {
                int index = map[i];
                int num = nout[index]++;
                int b = (index / BSIZE); index -= b*BSIZE;
                mapT[(maxNOut*b+num)*BSIZE+index] = i;
            }
        }
    }
};

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::postInit();

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

//////// CudaVec3f1

template <>
class SubsetMappingInternalData<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types> : public SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>
{
};

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::postInit();

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

template <>
class SubsetMappingInternalData<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes> : public SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>
{
};

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::postInit();

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

template <>
class SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types> : public SubsetMappingInternalData<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>
{
};

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::postInit();

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void SubsetMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
