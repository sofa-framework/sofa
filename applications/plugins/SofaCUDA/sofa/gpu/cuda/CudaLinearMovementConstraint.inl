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
#ifndef SOFA_GPU_CUDA_CUDALINEARMOVEMENTCONSTRAINT_INL
#define SOFA_GPU_CUDA_CUDALINEARMOVEMENTCONSTRAINT_INL

#include "CudaLinearMovementConstraint.h"
#include <SofaBoundaryCondition/LinearMovementConstraint.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void LinearMovementConstraintCudaVec6f_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaVec6f_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaVec6f_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);
    void LinearMovementConstraintCudaRigid3f_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaRigid3f_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaRigid3f_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE
    void LinearMovementConstraintCudaVec6d_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaVec6d_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaVec6d_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);
    void LinearMovementConstraintCudaRigid3d_projectResponseIndexed(unsigned size, const void* indices, void* dx);
    void LinearMovementConstraintCudaRigid3d_projectPositionIndexed(unsigned size, const void* indices, const void* dir, const void* x0, void* x);
    void LinearMovementConstraintCudaRigid3d_projectVelocityIndexed(unsigned size, const void* indices, const void* dir, void* dx);
#endif // SOFA_GPU_CUDA_DOUBLE

}// extern "C"

template<>
class CudaKernelsLinearMovementConstraint< CudaVec6fTypes >
{
public:
    static void projectResponse(unsigned size, const void* indices, void* dx)
    {
        LinearMovementConstraintCudaVec6f_projectResponseIndexed(size, indices, dx);
    }
    static void projectPosition(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
    {
        LinearMovementConstraintCudaVec6f_projectPositionIndexed(size, indices, dir, x0, x);
    }
    static void projectVelocity(unsigned size, const void* indices, const void* dir, void* dx)
    {
        LinearMovementConstraintCudaVec6f_projectVelocityIndexed(size, indices, dir, dx);
    }
};// CudaKernelsLinearMovementConstraint

template<>
class CudaKernelsLinearMovementConstraint< CudaRigid3fTypes >
{
public:
    static void projectResponse(unsigned size, const void* indices, void* dx)
    {
        LinearMovementConstraintCudaRigid3f_projectResponseIndexed(size, indices, dx);
    }
    static void projectPosition(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
    {
        LinearMovementConstraintCudaRigid3f_projectPositionIndexed(size, indices, dir, x0, x);
    }
    static void projectVelocity(unsigned size, const void* indices, const void* dir, void* dx)
    {
        LinearMovementConstraintCudaRigid3f_projectVelocityIndexed(size, indices, dir, dx);
    }
};// CudaKernelsLinearMovementConstraint

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsLinearMovementConstraint< CudaVec6dTypes >
{
public:
    static void projectResponse(unsigned size, const void* indices, void* dx)
    {
        LinearMovementConstraintCudaVec6d_projectResponseIndexed(size, indices, dx);
    }
    static void projectPosition(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
    {
        LinearMovementConstraintCudaVec6d_projectPositionIndexed(size, indices, dir, x0, x);
    }
    static void projectVelocity(unsigned size, const void* indices, const void* dir, void* dx)
    {
        LinearMovementConstraintCudaVec6d_projectVelocityIndexed(size, indices, dir, dx);
    }
};// CudaKernelsLinearMovementConstraint

template<>
class CudaKernelsLinearMovementConstraint< CudaRigid3dTypes >
{
public:
    static void projectResponse(unsigned size, const void* indices, void* dx)
    {
        LinearMovementConstraintCudaRigid3d_projectResponseIndexed(size, indices, dx);
    }
    static void projectPosition(unsigned size, const void* indices, const void* dir, const void* x0, void* x)
    {
        LinearMovementConstraintCudaRigid3d_projectPositionIndexed(size, indices, dir, x0, x);
    }
    static void projectVelocity(unsigned size, const void* indices, const void* dir, void* dx)
    {
        LinearMovementConstraintCudaRigid3d_projectVelocityIndexed(size, indices, dir, dx);
    }
};// CudaKernelsLinearMovementConstraint

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace cuda

} // namespace gpu

namespace component
{

namespace projectiveconstraintset
{

/////////////////////////////////////
// CudaVectorTypes specializations
/////////////////////////////////////
template<class TCoord, class TDeriv, class TReal>
void LinearMovementConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addIndex(Main* m, unsigned index)
{
    Data& data = *m->data;

    m->m_indices.beginEdit()->push_back(index);
    m->m_indices.endEdit();

    data.indices.push_back(index);
    // TODO : then it becomes non-consistent and also in the main version !!!
//  data.x0.push_back();

}// LinearMovementConstraintInternalData::addIndex

template<class TCoord, class TDeriv, class TReal>
void LinearMovementConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::removeIndex(Main* m, unsigned index)
{
    // Data& data = m->data;

    removeValue(*m->m_indices.beginEdit(),index);
    m->m_indices.endEdit();

    // removeValue(data.indices, index);
    // TODO : then it becomes non-consistent and also in the main version !!!
//  data.x0.push_back();

}// LinearMovementConstraintInternalData::removeIndex

template<class TCoord, class TDeriv, class TReal>
void LinearMovementConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m, VecCoord& x)
{
    Data& data = *m->data;
    const SetIndexArray & indices = m->m_indices.getValue();
//  m->x0.resize( indices.size() );
//  for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
//    m->x0[*it] = x[*it];

    // gpu part
    data.size = indices.size();
    data.indices.resize(data.size);
    unsigned index =0;
    for (typename SetIndex::const_iterator it = indices.begin(); it != indices.end(); it++)
    {
        data.indices[index] = *it;
        index++;
    }

    data.x0.resize(data.size);
    index = 0;
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        data.x0[index] = x[*it];
        index++;
    }
}// LinearMovementConstraintInternalData::init

template<class TCoord, class TDeriv, class TReal>
void LinearMovementConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    if ((cT != m->currentTime) || !m->finished)
    {
        m->findKeyTimes();
    }

    if (m->finished && m->nextT != m->prevT)
    {
        Kernels::projectResponse(
            data.size,
            data.indices.deviceRead(),
            dx.deviceWrite()
        );
    }
}// LinearMovementConstraintInternalData::projectResponse

template<class TCoord, class TDeriv, class TReal>
void LinearMovementConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::projectPosition(Main* m, VecCoord& x)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    // initialize initial Dofs positions, if it's not done
    if (data.x0.size() == 0)
    {
        data.init(m, x);
    }

    if ((cT != m->currentTime) || !m->finished)
    {
        m->findKeyTimes();
    }

    if (m->finished && m->nextT != m->prevT)
    {
        Real dt = (cT - m->prevT) / (m->nextT - m->prevT);
        Deriv depl = m->prevM + (m->nextM-m->prevM)*dt;

        Kernels::projectPosition(
            data.size,
            data.indices.deviceRead(),
            depl.ptr(),
            data.x0.deviceRead(),
            x.deviceWrite()
        );
    }
}// LinearMovementConstraintInternalData::projectPosition

template<class TCoord, class TDeriv, class TReal>
void LinearMovementConstraintInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::projectVelocity(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    if ((cT != m->currentTime) || !m->finished)
        m->findKeyTimes();

    if (m->finished && m->nextT != m->prevT)
    {
        Deriv dv = (m->nextM - m->prevM)*(1.0/(m->nextT - m->prevT));

        Kernels::projectVelocity(
            data.size,
            data.indices.deviceRead(),
            dv.ptr(),
            dx.deviceWrite()
        );
    }
}// LinearMovementConstraintInternalData::projectVelocity


/////////////////////////////////////
// CudaRigidTypes specializations
/////////////////////////////////////

template<int N, class real>
void LinearMovementConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::addIndex(Main* m, unsigned index)
{
    Data& data = *m->data;

    m->m_indices.beginEdit()->push_back(index);
    m->m_indices.endEdit();

    data.indices.push_back(index);
    // TODO : then it becomes non-consistent and also in the main version !!!
//  data.x0.push_back();

}// LinearMovementConstraintInternalData::addIndex

template<int N, class real>
void LinearMovementConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::removeIndex(Main* m, unsigned index)
{
    // Data& data = m->data;

    removeValue(*m->m_indices.beginEdit(),index);
    m->m_indices.endEdit();

    // removeValue(data.indices, index);
    // TODO : then it becomes non-consistent and also in the main version !!!
//  data.x0.push_back();

}// LinearMovementConstraintInternalData::removeIndex

template<int N, class real>
void LinearMovementConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::init(Main* m, VecCoord& x)
{
    Data& data = *m->data;
    const SetIndexArray & indices = m->m_indices.getValue();
//  m->x0.resize( indices.size() );
//  for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
//    m->x0[*it] = x[*it];

    // gpu part
    data.size = indices.size();
    data.indices.resize(data.size);
    unsigned index =0;
    for (typename SetIndex::const_iterator it = indices.begin(); it != indices.end(); it++)
    {
        data.indices[index] = *it;
        index++;
    }

    data.x0.resize(data.size);
    index = 0;
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        data.x0[index] = x[*it];
        index++;
    }
}// LinearMovementConstraintInternalData::init

template<int N, class real>
void LinearMovementConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::projectResponse(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    if ((cT != m->currentTime) || !m->finished)
    {
        m->findKeyTimes();
    }

    if (m->finished && m->nextT != m->prevT)
    {
        Kernels::projectResponse(
            data.size,
            data.indices.deviceRead(),
            dx.deviceWrite()
        );
    }
}// LinearMovementConstraintInternalData::projectResponse

template<int N, class real>
void LinearMovementConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::projectPosition(Main* m, VecCoord& x)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    // initialize initial Dofs positions, if it's not done
    if (data.x0.size() == 0)
    {
        data.init(m, x);
    }

    if ((cT != m->currentTime) || !m->finished)
    {
        m->findKeyTimes();
    }

    if (m->finished && m->nextT != m->prevT)
    {
        Real dt = (cT - m->prevT) / (m->nextT - m->prevT);
        Deriv depl = m->prevM + (m->nextM-m->prevM)*dt;

        Kernels::projectPosition(
            data.size,
            data.indices.deviceRead(),
            depl.ptr(),
            data.x0.deviceRead(),
            x.deviceWrite()
        );
    }
}// LinearMovementConstraintInternalData::projectPosition

template<int N, class real>
void LinearMovementConstraintInternalData< gpu::cuda::CudaRigidTypes<N, real> >::projectVelocity(Main* m, VecDeriv& dx)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    if ((cT != m->currentTime) || !m->finished)
        m->findKeyTimes();

    if (m->finished && m->nextT != m->prevT)
    {
        Deriv dv = (m->nextM - m->prevM)*(1.0/(m->nextT - m->prevT));

        Kernels::projectVelocity(
            data.size,
            data.indices.deviceRead(),
            dv.ptr(),
            dx.deviceWrite()
        );
    }
}// LinearMovementConstraintInternalData::projectVelocity


//////////////////////////////
// Specializations
/////////////////////////////

// template<>
// void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::init()
// {
//   data.init(this);
// }

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6fTypes >::addIndex(unsigned int index)
{
    data->addIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6fTypes >::removeIndex(unsigned int index)
{
    data->removeIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6fTypes >::projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectResponse(this, _dx);
    dx.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6fTypes >::projectJacobianMatrix(const core::MechanicalParams* /* mparams */, DataMatrixDeriv& /*dx*/)
{
    /*  data.projectResponseT(this, index); */
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6fTypes >::projectPosition(const core::MechanicalParams* /* mparams */, DataVecCoord& x)
{
    VecCoord& _x = *x.beginEdit();
    data->projectPosition(this, _x);
    x.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6fTypes >::projectVelocity(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectVelocity(this, _dx);
    dx.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::addIndex(unsigned int index)
{
    data->addIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::removeIndex(unsigned int index)
{
    data->removeIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectResponse(this, _dx);
    dx.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::projectJacobianMatrix(const core::MechanicalParams* /* mparams */, DataMatrixDeriv& /*dx*/)
{
    /*  data.projectResponseT(this, index); */
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::projectPosition(const core::MechanicalParams* /* mparams */, DataVecCoord& x)
{
    VecCoord& _x = *x.beginEdit();
    data->projectPosition(this, _x);
    x.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3fTypes >::projectVelocity(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectVelocity(this, _dx);
    dx.endEdit();
}

#ifdef SOFA_GPU_CUDA_DOUBLE
// template<>
// void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::init()
// {
//   data->init(this);
// }

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6dTypes >::addIndex(unsigned int index)
{
    data->addIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6dTypes >::removeIndex(unsigned int index)
{
    data->removeIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6dTypes >::projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectResponse(this, _dx);
    dx.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6dTypes >::projectJacobianMatrix(const core::MechanicalParams* /* mparams */, DataMatrixDeriv& /*dx*/)
{
    /*  data.projectResponseT(this, index); */
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6dTypes >::projectPosition(const core::MechanicalParams* /* mparams */, DataVecCoord& x)
{
    VecCoord& _x = *x.beginEdit();
    data->projectPosition(this, _x);
    x.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaVec6dTypes >::projectVelocity(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectVelocity(this, _dx);
    dx.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::addIndex(unsigned int index)
{
    data->addIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::removeIndex(unsigned int index)
{
    data->removeIndex(this, index);
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::projectResponse(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectResponse(this, _dx);
    dx.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::projectJacobianMatrix(const core::MechanicalParams* /* mparams */, DataMatrixDeriv& /*dx*/)
{
    /*  data.projectResponseT(this, index); */
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::projectPosition(const core::MechanicalParams* /* mparams */, DataVecCoord& x)
{
    VecCoord& _x = *x.beginEdit();
    data->projectPosition(this, _x);
    x.endEdit();
}

template<>
void LinearMovementConstraint< gpu::cuda::CudaRigid3dTypes >::projectVelocity(const core::MechanicalParams* /* mparams */, DataVecDeriv& dx)
{
    VecDeriv& _dx = *dx.beginEdit();
    data->projectVelocity(this, _dx);
    dx.endEdit();
}

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
