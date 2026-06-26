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
#include <sofa/component/constraint/projective/LinearMovementProjectiveConstraint.h>

namespace sofa::gpu::cuda
{

template<class DataTypes>
class CudaKernelsLinearMovementProjectiveConstraint;

}// namespace sofa::gpu::cuda

namespace sofa::component::constraint::projective
{

template<class TCoord, class TDeriv, class TReal>
class LinearMovementProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >
{
public:
    using Index = sofa::Index;
    typedef LinearMovementProjectiveConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> > Data;
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef LinearMovementProjectiveConstraint<DataTypes> Main;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef type::vector<Index> SetIndexArray;
    typedef sofa::core::topology::PointSubsetData< SetIndexArray > SetIndex;

    typedef sofa::core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef sofa::core::objectmodel::Data<MatrixDeriv> DataMatrixDeriv;

    typedef gpu::cuda::CudaKernelsLinearMovementProjectiveConstraint<DataTypes> Kernels;

    // vector of indices for general case
    gpu::cuda::CudaVector<int> indices;

    // initial positions
    gpu::cuda::CudaVector<Coord> x0;

    int size;

    static void init(Main* m, VecCoord& x);

    static void addIndex(Main* m, Index index);

    static void removeIndex(Main* m, Index index);

    static void projectResponse(Main* m, VecDeriv& dx);
    static void projectPosition(Main* m, VecCoord& x);
    static void projectVelocity(Main* m, VecDeriv& dx);

};


template<int N, class real>
class LinearMovementProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >
{
public:
    using Index = sofa::Index;
    typedef LinearMovementProjectiveConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> > Data;
    typedef gpu::cuda::CudaRigidTypes<N, real> DataTypes;
    typedef LinearMovementProjectiveConstraint<DataTypes> Main;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef type::vector<Index> SetIndexArray;
    typedef sofa::core::topology::PointSubsetData< SetIndexArray > SetIndex;

    typedef sofa::core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef sofa::core::objectmodel::Data<MatrixDeriv> DataMatrixDeriv;

    typedef gpu::cuda::CudaKernelsLinearMovementProjectiveConstraint<DataTypes> Kernels;

    // vector of indices for general case
    gpu::cuda::CudaVector<int> indices;

    // initial positions
    gpu::cuda::CudaVector<Coord> x0;

    int size;

    static void init(Main* m, VecCoord& x);

    static void addIndex(Main* m, Index index);

    static void removeIndex(Main* m, Index index);

    static void projectResponse(Main* m, VecDeriv& dx);
    static void projectPosition(Main* m, VecCoord& x);
    static void projectVelocity(Main* m, VecDeriv& dx);

};


// template<>
// void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::init();

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6fTypes >::addIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6fTypes >::removeIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6fTypes >::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6fTypes >::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6fTypes >::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& x);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6fTypes >::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::addIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::removeIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& x);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3fTypes >::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

// template<>
// void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::init();

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6dTypes >::addIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6dTypes >::removeIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6dTypes >::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6dTypes >::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6dTypes >::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& x);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaVec6dTypes >::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::addIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::removeIndex(Index index);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& dx);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& x);

template<>
void LinearMovementProjectiveConstraint< gpu::cuda::CudaRigid3dTypes >::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& dx);

#endif // SOFA_GPU_CUDA_DOUBLE



} // namespace sofa::component::constraint::projective
