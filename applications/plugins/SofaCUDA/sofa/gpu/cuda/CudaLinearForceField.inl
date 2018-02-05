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
#include "CudaTypes.h"
#include "CudaLinearForceField.h"
#include <SofaBoundaryCondition/LinearForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void LinearForceFieldCudaRigid3f_addForce(unsigned size, const void* indices, const void* forces, void* f);
#ifdef SOFA_GPU_CUDA_DOUBLE
    void LinearForceFieldCudaRigid3d_addForce(unsigned size, const void* indices, const void *forces, void* f);
#endif
}

template<>
class CudaKernelsLinearForceField< CudaRigid3fTypes >
{
public:
    static void addForce(unsigned size, const void* indices, const void* forces, void* f)
	{
        LinearForceFieldCudaRigid3f_addForce(size, indices, forces, f);
    }
}; //CudaKernelsLinearForceField< CudaRigid3fTypes >

#ifdef SOFA_GPU_CUDA_DOUBLE
template<>
class CudaKernelsLinearForceField< CudaRigid3dTypes >
{
public:
    static void addForce(unsigned size, const void* indices, const void* forces, void* f)
    {
        LinearForceFieldCudaRigid3d_addForce(size, indices, forces, f);
    }
}; //CudaKernelsLinearForceField< CudaRigid3dTypes >
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace cuda

} // namespace gpu


namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template<int N, class real>
void LinearForceFieldInternalData< gpu::cuda::CudaRigidTypes<N, real> >::init(Main* m)
{
    Data& data = *m->data;

    data.indices.clear();

    const SetIndexArray& m_indices = m->points.getValue();

    data.indices.resize(m_indices.size());
	data.size = data.indices.size();

    for(unsigned i = 0; i < m_indices.size(); i++)
        data.indices[i] = m_indices[i];

}// LinearForceFieldInternalData::init

template<int N, class real>
void LinearForceFieldInternalData< gpu::cuda::CudaRigidTypes<N, real> >::addForce(Main* m, VecDeriv& f)
{
    Data& data = *m->data;

    Real cT = (Real) m->getContext()->getTime();

    if (m->d_keyTimes.getValue().size() != 0 && cT >= *m->d_keyTimes.getValue().begin() && cT <= *m->d_keyTimes.getValue().rbegin())
    {
        m->nextT = *m->d_keyTimes.getValue().begin();
        m->prevT = m->nextT;

        bool finished = false;

        typename helper::vector< Real >::const_iterator it_t = m->d_keyTimes.getValue().begin();
        typename VecDeriv::const_iterator it_f = m->d_keyForces.getValue().begin();

        // WARNING : we consider that the key-events are in chronological order
        // here we search between which keyTimes we are.
        while( it_t != m->d_keyTimes.getValue().end() && !finished)
        {
            if ( *it_t <= cT)
            {
                m->prevT = *it_t;
                m->prevF = *it_f;
            }
            else
            {
                m->nextT = *it_t;
                m->nextF = *it_f;
                finished = true;
            }
            it_t++;
            it_f++;
        }

        if (finished)
        {
            Deriv slope = (m->nextF - m->prevF)*(1.0/(m->nextT - m->prevT));
            Deriv ff = slope*(cT - m->prevT) + m->prevF;
            ff = ff*m->d_force.getValue();

            Kernels::addForce(
                data.size,
                data.indices.deviceRead(),
                ff.ptr(),
                f.deviceWrite()
            );
        }
    }
}// LinearForceFieldInternalData::addForce

template<>
void LinearForceField<sofa::gpu::cuda::CudaRigid3fTypes>::init()
{
    data->init(this);
	Inherit::init();
}// LinearForceFieldInternalData::init

template<>
void LinearForceField<sofa::gpu::cuda::CudaRigid3fTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& /*p*/, const DataVecDeriv& /*v*/)
{
    VecDeriv& _f = *f.beginEdit();
    data->addForce(this, _f);
    f.endEdit();
}// LinearForceField::addForce

template<>
SReal LinearForceField<sofa::gpu::cuda::CudaRigid3fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    this->serr<<"["<<this->getName()<<"] getPotentialEnergy not implemented !"<<this->sendl;
    return 0;
}

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
void LinearForceField<sofa::gpu::cuda::CudaRigid3dTypes>::init()
{
    data->init(this);
}// LinearForceFieldInternalData::init

template<>
void LinearForceField<sofa::gpu::cuda::CudaRigid3dTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& /*p*/, const DataVecDeriv& /*v*/)
{
    VecDeriv& _f = *f.beginEdit();
    data->addForce(this, _f);
    f.endEdit();
}// LinearForceField::addForce

template<>
SReal LinearForceField<sofa::gpu::cuda::CudaRigid3dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    this->serr<<"["<<this->getName()<<"] getPotentialEnergy not implemented !"<<this->sendl;
    return 0;
}

#endif // SOFA_GPU_CUDA_DOUBLE

}// namespace forcefield

}// namespace component

}// namespace sofa
