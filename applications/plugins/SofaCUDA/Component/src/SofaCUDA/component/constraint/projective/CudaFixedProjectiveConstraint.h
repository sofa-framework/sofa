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

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/constraint/projective/FixedProjectiveConstraint.h>

namespace sofa::component::constraint::projective
{

template<class TCoord, class TDeriv, class TReal>
class FixedProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    using Index = sofa::Index;
    typedef FixedProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > Data;
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef FixedProjectiveConstraint<DataTypes> Main;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename Main::SetIndex SetIndex;
    typedef typename Main::SetIndexArray SetIndexArray;

    // min/max fixed indices for contiguous constraints
    Index minIndex;
    Index maxIndex;
    // vector of indices for general case
    gpu::cuda::CudaVector<int> cudaIndices;


    static void init(Main* m);

    static void addConstraint(Main* m, Index index);

    static void removeConstraint(Main* m, Index index);

    static void projectResponse(Main* m, VecDeriv& dx);
};

template <int N, class real>
class FixedProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real > >
{
public:
    using Index = sofa::Index;
    typedef FixedProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> > Data;
    typedef gpu::cuda::CudaRigidTypes<N, real> DataTypes;
    typedef FixedProjectiveConstraint<DataTypes> Main;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename Main::SetIndex SetIndex;
    typedef typename Main::SetIndexArray SetIndexArray;

    // min/max fixed indices for contiguous constraints
    Index minIndex;
    Index maxIndex;
    // vector of indices for general case
    gpu::cuda::CudaVector<int> cudaIndices;


    static void init(Main* m);

    static void addConstraint(Main* m, Index index);

    static void removeConstraint(Main* m, Index index);

    static void projectResponse(Main* m, VecDeriv& dx);
};

// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaFixedProjectiveConstraint_DeclMethods(T) \
    template<> void FixedProjectiveConstraint< T >::init(); \
    template<> void FixedProjectiveConstraint< T >::addConstraint(Index index); \
    template<> void FixedProjectiveConstraint< T >::removeConstraint(Index index); \
    template<> void FixedProjectiveConstraint< T >::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData);

CudaFixedProjectiveConstraint_DeclMethods(gpu::cuda::CudaVec3fTypes);
CudaFixedProjectiveConstraint_DeclMethods(gpu::cuda::CudaVec3f1Types);
CudaFixedProjectiveConstraint_DeclMethods(gpu::cuda::CudaRigid3fTypes);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaFixedProjectiveConstraint_DeclMethods(gpu::cuda::CudaVec3dTypes);
CudaFixedProjectiveConstraint_DeclMethods(gpu::cuda::CudaVec3d1Types);
CudaFixedProjectiveConstraint_DeclMethods(gpu::cuda::CudaRigid3dTypes);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaFixedProjectiveConstraint_DeclMethods


} // namespace sofa::component::constraint::projective
