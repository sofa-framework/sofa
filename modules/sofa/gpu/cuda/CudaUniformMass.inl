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
#ifndef SOFA_GPU_CUDA_CUDAUNIFORMMASS_INL
#define SOFA_GPU_CUDA_CUDAUNIFORMMASS_INL

#include "CudaUniformMass.h"
#include <sofa/component/mass/UniformMass.inl>
#include <sofa/helper/gl/Axis.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);

    void UniformMassCuda3f1_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f1_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f1_addForce(unsigned int size, const float *mg, void* f);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void UniformMassCuda3d_addMDx(unsigned int size, double mass, void* res, const void* dx);
    void UniformMassCuda3d_accFromF(unsigned int size, double mass, void* a, const void* f);
    void UniformMassCuda3d_addForce(unsigned int size, const double *mg, void* f);

    void UniformMassCuda3d1_addMDx(unsigned int size, double mass, void* res, const void* dx);
    void UniformMassCuda3d1_accFromF(unsigned int size, double mass, void* a, const void* f);
    void UniformMassCuda3d1_addForce(unsigned int size, const double *mg, void* f);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mass
{

using namespace gpu::cuda;

// -- Mass interface
template <>
void UniformMass<CudaVec3fTypes, float>::addMDx(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecDeriv& d_dx, double d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3f_addMDx(dx.size(), (float)(mass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3fTypes, float>::accFromF(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3f_accFromF(f.size(), mass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3fTypes, float>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass.getValue();
    UniformMassCuda3f_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3f1Types, float>::addMDx(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecDeriv& d_dx, double d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3f1_addMDx(dx.size(), (float)(mass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3f1Types, float>::accFromF(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3f1_accFromF(f.size(), mass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3f1Types, float>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass.getValue();
    UniformMassCuda3f1_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord& d_x) const
{
    const VecCoord& x = d_x.getValue();

    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
double UniformMass<gpu::cuda::CudaRigid3fTypes,sofa::defaulttype::Rigid3fMass>::getElementMass(unsigned int ) const
{
    return (double)(mass.getValue().mass);
}

template <>
void UniformMass<gpu::cuda::CudaRigid3fTypes, Rigid3fMass>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    defaulttype::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = mass.getValue().inertiaMatrix[0][0];
    double m11 = mass.getValue().inertiaMatrix[1][1];
    double m22 = mass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len);
    }
}


#ifdef SOFA_GPU_CUDA_DOUBLE

// -- Mass interface
template <>
void UniformMass<CudaVec3dTypes, double>::addMDx(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecDeriv& d_dx, double d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3d_addMDx(dx.size(), (double)(mass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3dTypes, double>::accFromF(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3d_accFromF(f.size(), mass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3dTypes, double>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass.getValue();
    UniformMassCuda3d_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

template <>
bool UniformMass<gpu::cuda::CudaVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mstate->getX();
    //if (!x.isHostValid()) return false; // Do not recompute bounding box if it requires to transfer data from device
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        const Coord& p = x.getCached(i);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

template <>
void UniformMass<CudaVec3d1Types, double>::addMDx(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecDeriv& d_dx, double d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3d1_addMDx(dx.size(), (double)(mass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3d1Types, double>::accFromF(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3d1_accFromF(f.size(), mass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3d1Types, double>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * mass.getValue();
    UniformMassCuda3d1_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

template <>
bool UniformMass<gpu::cuda::CudaVec3d1Types, double>::addBBox(double* minBBox, double* maxBBox)
{
    const VecCoord& x = *this->mstate->getX();
    //if (!x.isHostValid()) return false; // Do not recompute bounding box if it requires to transfer data from device
    for (unsigned int i=0; i<x.size(); i++)
    {
        //const Coord& p = x[i];
        const Coord& p = x.getCached(i);
        for (int c=0; c<3; c++)
        {
            if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
            if (p[c] < minBBox[c]) minBBox[c] = p[c];
        }
    }
    return true;
}

template <>
double UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord& d_x) const
{
    const VecCoord& x = d_x.getValue();

    double e = 0;
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += g*mass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
double UniformMass<gpu::cuda::CudaRigid3dTypes,sofa::defaulttype::Rigid3dMass>::getElementMass(unsigned int ) const
{
    return (double)(mass.getValue().mass);
}

template <>
void UniformMass<gpu::cuda::CudaRigid3dTypes, Rigid3dMass>::draw(const core::visual::VisualParams* vparams )
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = *mstate->getX();
    defaulttype::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = mass.getValue().inertiaMatrix[0][0];
    double m11 = mass.getValue().inertiaMatrix[1][1];
    double m22 = mass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        helper::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len);
    }
}

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace mass

} // namespace component

} // namespace sofa

#endif
