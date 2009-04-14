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
#ifndef SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL

#include "CudaPlaneForceField.h"
#include <sofa/component/forcefield/PlaneForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class real>
struct PlaneDForceOp
{
    unsigned int size;
    GPUPlane<real> plane;
    const void* penetration;
    void* f;
    const void* dx;
};

extern "C"
{

    void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane<float>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane<float>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void MultiPlaneForceFieldCuda3f_addDForce(int n, PlaneDForceOp<float>* ops);

    void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane<float>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane<float>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void PlaneForceFieldCuda3d_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3d1_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d1_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

#endif // SOFA_GPU_CUDA_DOUBLE

}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template <>
bool PlaneForceField<gpu::cuda::CudaVec3fTypes>::canPrefetch() const
{
    return mycudaMultiOpMax != 0;
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (this->isPrefetching()) return;
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3f_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    if (this->isPrefetching())
    {
        PlaneDForceOp<float> op;
        op.size = dx.size();
        op.plane = data.plane;
        op.plane.stiffness *= (Real)kFactor;
        op.penetration = data.penetration.deviceRead();
        op.f = df.deviceWrite();
        op.dx = dx.deviceRead();

        data.preDForceOpID = data.opsDForce().size();
        data.opsDForce().push_back(op);
        return;
    }
    else if (data.preDForceOpID != -1)
    {
        helper::vector<PlaneDForceOp<float> >& ops = data.opsDForce();
        if (!ops.empty())
        {
            if (ops.size() == 1)
            {
                // only one object -> use regular kernel
                data.preDForceOpID = -1;
            }
            else
            {
                MultiPlaneForceFieldCuda3f_addDForce(ops.size(), &(ops[0]));
            }
            ops.clear();
        }
        if (data.preDForceOpID != -1)
        {
            data.preDForceOpID = -1;
            return;
        }
    }
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3f_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = (Real)stiff;
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3f1_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3f1_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = (Real)stiff;
}

#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3d_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3d_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = (Real)stiff;
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3d1_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3d1_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = (Real)stiff;
}

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
