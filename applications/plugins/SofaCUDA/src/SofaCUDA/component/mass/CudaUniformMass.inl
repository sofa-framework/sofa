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
#ifndef SOFA_GPU_CUDA_CUDAUNIFORMMASS_INL
#define SOFA_GPU_CUDA_CUDAUNIFORMMASS_INL

#include <SofaCUDA/component/mass/CudaUniformMass.h>
#include <sofa/component/mass/UniformMass.inl>
#include <sofa/gl/Axis.h>

namespace sofa
{


namespace gpu::cuda
{

extern "C"
{
    void UniformMassCuda3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f_addForce(unsigned int size, const float *mg, void* f);

    void UniformMassCuda3f1_addMDx(unsigned int size, float mass, void* res, const void* dx);
    void UniformMassCuda3f1_accFromF(unsigned int size, float mass, void* a, const void* f);
    void UniformMassCuda3f1_addForce(unsigned int size, const float *mg, void* f);

        void UniformMassCudaRigid3f_addMDx(unsigned int size, float mass, void* res, const void* dx);
        void UniformMassCudaRigid3f_accFromF(unsigned int size, float mass, void* a, const void* dx);
        void UniformMassCudaRigid3f_addForce(unsigned int size, const float* mg, void* f);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void UniformMassCuda3d_addMDx(unsigned int size, double mass, void* res, const void* dx);
    void UniformMassCuda3d_accFromF(unsigned int size, double mass, void* a, const void* f);
    void UniformMassCuda3d_addForce(unsigned int size, const double *mg, void* f);

    void UniformMassCuda3d1_addMDx(unsigned int size, double mass, void* res, const void* dx);
    void UniformMassCuda3d1_accFromF(unsigned int size, double mass, void* a, const void* f);
    void UniformMassCuda3d1_addForce(unsigned int size, const double *mg, void* f);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace gpu::cuda


namespace component::mass
{

using namespace gpu::cuda;

// -- Mass interface
template <>
void UniformMass<CudaVec3fTypes>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecDeriv& d_dx, SReal d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3f_addMDx(dx.size(), (float)(d_vertexMass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3fTypes>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3f_accFromF(f.size(), d_vertexMass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * d_vertexMass.getValue();
    UniformMassCuda3f_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3f1Types>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecDeriv& d_dx, SReal d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3f1_addMDx(dx.size(), (float)(d_vertexMass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3f1Types>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3f1_accFromF(f.size(), d_vertexMass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3f1Types>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    type::Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * d_vertexMass.getValue();
    UniformMassCuda3f1_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes>::addMDx(const core::MechanicalParams * /*mparams*/, DataVecDeriv &f, const DataVecDeriv &dx, SReal factor)
{
        VecDeriv& _f = *f.beginEdit();
        const VecDeriv& _dx = dx.getValue();


        UniformMassCudaRigid3f_addMDx(_dx.size(), (float)(d_vertexMass.getValue().mass*factor), _f.deviceWrite(), _dx.deviceRead());

//	for(int i = 0 ; i < _f.size() ; ++i)
//		std::cout << "CPU "<< i << "  : " << _f[i] << std::endl;

        f.endEdit();
}

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes>::accFromF(const core::MechanicalParams * /*mparams*/, DataVecDeriv &a, const DataVecDeriv &f)
{
        VecDeriv& _a = *a.beginEdit();
        const VecDeriv _f = f.getValue();

        UniformMassCudaRigid3f_accFromF(_a.size(), d_vertexMass.getValue().mass, _a.deviceWrite(), _f.deviceRead());

        a.endEdit();
}

template<>
void UniformMass<gpu::cuda::CudaRigid3fTypes>::addForce(const core::MechanicalParams * /*mparams*/, DataVecDeriv &f, const DataVecCoord& /*x*/, const DataVecDeriv& /*v*/)
{

        VecDeriv& _f = *f.beginEdit();
        type::Vec3d g(this->getContext()->getGravity());

        const float m = d_vertexMass.getValue().mass;
        const float mg[] = { (float)(m*g(0)), (float)(m*g(1)), (float)(m*g(2)) };
        UniformMassCudaRigid3f_addForce(_f.size(), mg, _f.deviceWrite());

        f.endEdit();

}


template <>
SReal UniformMass<gpu::cuda::CudaRigid3fTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& d_x) const
{
    const VecCoord& x = d_x.getValue();

    SReal e = 0;
    // gravity
    const type::Vec3d g ( this->getContext()->getGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += g*d_vertexMass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
SReal UniformMass<gpu::cuda::CudaRigid3fTypes>::getElementMass(sofa::Index) const
{
    return (SReal)(d_vertexMass.getValue().mass);
}

template <>
void UniformMass<gpu::cuda::CudaRigid3fTypes>::draw(const core::visual::VisualParams* vparams)
{
#if SOFACUDA_HAVE_SOFA_GL == 1
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = mstate->read(core::vec_id::read_access::position)->getValue();
    type::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    const double m00 = d_vertexMass.getValue().inertiaMatrix[0][0];
    const double m11 = d_vertexMass.getValue().inertiaMatrix[1][1];
    const double m22 = d_vertexMass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        sofa::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len, sofa::type::RGBAColor::red(), sofa::type::RGBAColor::green(), sofa::type::RGBAColor::blue());
    }
#endif // SOFACUDA_HAVE_SOFA_GL == 1
}


#ifdef SOFA_GPU_CUDA_DOUBLE

// -- Mass interface
template <>
void UniformMass<CudaVec3dTypes>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecDeriv& d_dx, SReal d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3d_addMDx(dx.size(), (double)(d_vertexMass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3dTypes>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3d_accFromF(f.size(), d_vertexMass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * d_vertexMass.getValue();
    UniformMassCuda3d_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

// template <>
// bool UniformMass<gpu::cuda::CudaVec3dTypes, double>::addBBox(SReal* minBBox, SReal* maxBBox)
// {
//     const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
//     //if (!x.isHostValid()) return false; // Do not recompute bounding box if it requires to transfer data from device
//     for (unsigned int i=0; i<x.size(); i++)
//     {
//         //const Coord& p = x[i];
//         const Coord& p = x.getCached(i);
//         for (int c=0;c<3;c++)
//         {
//             if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
//             if (p[c] < minBBox[c]) minBBox[c] = p[c];
//         }
//     }
//     return true;
// }

template <>
void UniformMass<CudaVec3d1Types>::addMDx(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecDeriv& d_dx, SReal d_factor)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    UniformMassCuda3d1_addMDx(dx.size(), (double)(d_vertexMass.getValue()*d_factor), f.deviceWrite(), dx.deviceRead());

    d_f.endEdit();
}

template <>
void UniformMass<CudaVec3d1Types>::accFromF(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_a, const DataVecDeriv& d_f)
{
    VecDeriv& a = *d_a.beginEdit();
    const VecDeriv& f = d_f.getValue();

    UniformMassCuda3d1_accFromF(f.size(), d_vertexMass.getValue(), a.deviceWrite(), f.deviceRead());

    d_a.endEdit();
}

template <>
void UniformMass<CudaVec3d1Types>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& /*d_x*/, const DataVecDeriv& /*d_v*/)
{
    VecDeriv& f = *d_f.beginEdit();
    //const VecCoord& x = d_x.getValue();
    //const VecDeriv& v = d_v.getValue();

    // weight
    Vec3d g ( this->getContext()->getGravity() );
    Deriv theGravity;
    DataTypes::set( theGravity, g[0], g[1], g[2]);
    Deriv mg = theGravity * d_vertexMass.getValue();
    UniformMassCuda3d1_addForce(f.size(), mg.ptr(), f.deviceWrite());

    d_f.endEdit();
}

// template <>
// bool UniformMass<gpu::cuda::CudaVec3d1Types, double>::addBBox(double* minBBox, double* maxBBox)
// {
//     const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
//     //if (!x.isHostValid()) return false; // Do not recompute bounding box if it requires to transfer data from device
//     for (unsigned int i=0; i<x.size(); i++)
//     {
//         //const Coord& p = x[i];
//         const Coord& p = x.getCached(i);
//         for (int c=0;c<3;c++)
//         {
//             if (p[c] > maxBBox[c]) maxBBox[c] = p[c];
//             if (p[c] < minBBox[c]) minBBox[c] = p[c];
//         }
//     }
//     return true;
// }

template <>
SReal UniformMass<gpu::cuda::CudaRigid3dTypes>::getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& d_x) const
{
    const VecCoord& x = d_x.getValue();

    SReal e = 0;
    // gravity
    Vec3d g ( this->getContext()->getGravity() );
    for (unsigned int i=0; i<x.size(); i++)
    {
        e += g*d_vertexMass.getValue().mass*x[i].getCenter();
    }
    return e;
}

template <>
SReal UniformMass<gpu::cuda::CudaRigid3dTypes>::getElementMass(sofa::Index) const
{
    return (SReal)(d_vertexMass.getValue().mass);
}

template <>
void UniformMass<gpu::cuda::CudaRigid3dTypes>::draw(const core::visual::VisualParams* vparams )
{
    if (!vparams->displayFlags().getShowBehaviorModels())
        return;
    const VecCoord& x = mstate->read(core::vec_id::read_access::position)->getValue();
    type::Vec3d len;

    // The moment of inertia of a box is:
    //   m->_I(0,0) = M/REAL(12.0) * (ly*ly + lz*lz);
    //   m->_I(1,1) = M/REAL(12.0) * (lx*lx + lz*lz);
    //   m->_I(2,2) = M/REAL(12.0) * (lx*lx + ly*ly);
    // So to get lx,ly,lz back we need to do
    //   lx = sqrt(12/M * (m->_I(1,1)+m->_I(2,2)-m->_I(0,0)))
    // Note that RigidMass inertiaMatrix is already divided by M
    double m00 = d_vertexMass.getValue().inertiaMatrix[0][0];
    double m11 = d_vertexMass.getValue().inertiaMatrix[1][1];
    double m22 = d_vertexMass.getValue().inertiaMatrix[2][2];
    len[0] = sqrt(m11+m22-m00);
    len[1] = sqrt(m00+m22-m11);
    len[2] = sqrt(m00+m11-m22);

    for (unsigned int i=0; i<x.size(); i++)
    {
        sofa::gl::Axis::draw(x[i].getCenter(), x[i].getOrientation(), len, sofa::type::RGBAColor::red(), sofa::type::RGBAColor::green(), sofa::type::RGBAColor::blue());
    }
}

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace component::mass


} // namespace sofa

#endif
