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
#ifndef SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_INL

#include "CudaSpringForceField.h"
#include <SofaDeformable/SpringForceField.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <SofaDeformable/MeshSpringForceField.inl>
#include <SofaGeneralDeformable/TriangleBendingSprings.inl>
#include <SofaGeneralDeformable/QuadBendingSprings.inl>
#include <sofa/core/MechanicalParams.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SpringForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3f_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3f_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

    void SpringForceFieldCuda3f1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3f1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3f1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3f1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3f1_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3f1_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void SpringForceFieldCuda3d_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3d_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3d_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3d_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3d_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3d_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

    void SpringForceFieldCuda3d1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v);
    void SpringForceFieldCuda3d1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2);
    void StiffSpringForceFieldCuda3d1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx);
    void StiffSpringForceFieldCuda3d1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx);
    void StiffSpringForceFieldCuda3d1_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, double factor);
    void StiffSpringForceFieldCuda3d1_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor);

#endif // SOFA_GPU_CUDA_DOUBLE

} // extern "C"


template<>
class CudaKernelsSpringForceField<CudaVec3fTypes>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
    {   SpringForceFieldCuda3f_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
    {   SpringForceFieldCuda3f_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
    {   StiffSpringForceFieldCuda3f_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
    {   StiffSpringForceFieldCuda3f_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, SReal factor)
    {   StiffSpringForceFieldCuda3f_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
    {   StiffSpringForceFieldCuda3f_addExternalDForce(nbVertex, nbSpringPerVertex, springs, f1, dx1, x1, dx2, x2, dfdx, factor); }
};

template<>
class CudaKernelsSpringForceField<CudaVec3f1Types>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
    {   SpringForceFieldCuda3f1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
    {   SpringForceFieldCuda3f1_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
    {   StiffSpringForceFieldCuda3f1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
    {   StiffSpringForceFieldCuda3f1_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, SReal factor)
    {   StiffSpringForceFieldCuda3f1_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
    {   StiffSpringForceFieldCuda3f1_addExternalDForce(nbVertex, nbSpringPerVertex, springs, f1, dx1, x1, dx2, x2, dfdx, factor); }
};

#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
class CudaKernelsSpringForceField<CudaVec3dTypes>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
    {   SpringForceFieldCuda3d_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
    {   SpringForceFieldCuda3d_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
    {   StiffSpringForceFieldCuda3d_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
    {   StiffSpringForceFieldCuda3d_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, SReal factor)
    {   StiffSpringForceFieldCuda3d_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
    {   StiffSpringForceFieldCuda3d_addExternalDForce(nbVertex, nbSpringPerVertex, springs, f1, dx1, x1, dx2, x2, dfdx, factor); }
};

template<>
class CudaKernelsSpringForceField<CudaVec3d1Types>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v)
    {   SpringForceFieldCuda3d1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2)
    {   SpringForceFieldCuda3d1_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* x, const void* v, void* dfdx)
    {   StiffSpringForceFieldCuda3d1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* x1, const void* v1, const void* x2, const void* v2, void* dfdx)
    {   StiffSpringForceFieldCuda3d1_addExternalForce(nbVertex, nbSpringPerVertex, springs, f1, x1, v1, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f, const void* dx, const void* x, const void* dfdx, SReal factor)
    {   StiffSpringForceFieldCuda3d1_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const void* springs, void* f1, const void* dx1, const void* x1, const void* dx2, const void* x2, const void* dfdx, double factor)
    {   StiffSpringForceFieldCuda3d1_addExternalDForce(nbVertex, nbSpringPerVertex, springs, f1, dx1, x1, dx2, x2, dfdx, factor); }
};

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace cuda

} // namespace gpu

namespace component
{

namespace interactionforcefield
{

using namespace gpu::cuda;

template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m, bool stiff)
{
    Data& data = m->data;
    m->Inherit::init();
    const sofa::type::vector<Spring>& springs = m->springs.getValue();
    if (!springs.empty())
    {
        std::map<int,int> nsprings;
        for (unsigned int i=0; i<springs.size(); i++)
        {
            nsprings[springs[i].m1]++;
            nsprings[springs[i].m2]++;
        }

        int nmax = 0;
        for (std::map<int,int>::const_iterator it = nsprings.begin(); it != nsprings.end(); ++it)
            if (it->second > nmax)
                nmax = it->second;
        data.springs1.init(nsprings.begin()->first, nsprings.rbegin()->first - nsprings.begin()->first + 1, nmax);
        std::cout << "CUDA SpringForceField: "<<springs.size()<<" springs, "<<data.springs1.nbVertex<<" attached points, max "<<data.springs1.nbSpringPerVertex<<" springs per point."<<std::endl;
        nsprings.clear();
        for (unsigned int i=0; i<springs.size(); i++)
        {
            int m1 = springs[i].m1 - data.springs1.vertex0;
            int m2 = springs[i].m2 - data.springs1.vertex0;
            data.springs1.set(m1, nsprings[m1]++, m2,
                              (float)springs[i].initpos,
                              (float)springs[i].ks,
                              (float)springs[i].kd);
            data.springs1.set(m2, nsprings[m2]++, m1,
                              (float)springs[i].initpos,
                              (float)springs[i].ks,
                              (float)springs[i].kd);
        }
    }
    if (stiff)
    {
        data.springs1.dfdx.resize(data.springs1.springs.size());
        data.springs2.dfdx.resize(data.springs2.springs.size());
    }
}

// -- InteractionForceField interface
template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, bool stiff, VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    Data& data = m->data;

    f.resize(x.size());
    int d = data.springs1.vertex0;
    if (data.springs1.nbSpringPerVertex > 0)
    {
        if (!stiff)
            Kernels::addForce(data.springs1.nbVertex,
                              data.springs1.nbSpringPerVertex,
                              data.springs1.springs.deviceRead(),
                              (      Deriv*)f.deviceWrite() + d,
                              (const Coord*)x.deviceRead()  + d,
                              (const Deriv*)v.deviceRead()  + d);
        else
            Kernels::addForce(data.springs1.nbVertex,
                              data.springs1.nbSpringPerVertex,
                              data.springs1.springs.deviceRead(),
                              (      Deriv*)f.deviceWrite() + d,
                              (const Coord*)x.deviceRead()  + d,
                              (const Deriv*)v.deviceRead()  + d,
                              data.springs1.dfdx.deviceWrite());
    }
}

template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce(Main* m, bool stiff, VecDeriv& df, const VecDeriv& dx, SReal kFactor, SReal /*bFactor*/)
{
    if (!stiff) return;
    Data& data = m->data;
    const VecCoord& x = m->getMState()->read(core::ConstVecCoordId::position())->getValue();
    df.resize(x.size());
    int d = data.springs1.vertex0;
    if (data.springs1.nbSpringPerVertex > 0)
    {
        Kernels::addDForce(data.springs1.nbVertex,
                           data.springs1.nbSpringPerVertex,
                           data.springs1.springs.deviceRead(),
                           (      Deriv*)df.deviceWrite() + d,
                           (const Deriv*)dx.deviceRead() + d,
                           (const Coord*)x.deviceRead()  + d,
                           data.springs1.dfdx.deviceRead(),
                           kFactor);
    }
}


// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaSpringForceField_ImplMethods(T) \
    template<> void SpringForceField< T >::init() \
    { \
	    Inherit1::init();   \
        data.init(this, false); \
    } \
    template<> void SpringForceField< T >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) \
    { \
        VecDeriv& _f = *f.beginEdit(); \
        const VecCoord& _x = x.getValue(); \
        const VecDeriv& _v = v.getValue(); \
		data.addForce(this, false, _f, _x, _v);\
        f.endEdit(); \
	} \
    template<> void StiffSpringForceField< T >::init() \
    { \
	    Inherit1::init(); \
        data.init(this, true); \
    } \
    template<> void StiffSpringForceField< T >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) \
    { \
        VecDeriv& _f = *f.beginEdit(); \
        const VecCoord& _x = x.getValue(); \
        const VecDeriv& _v = v.getValue(); \
		data.addForce(this, true, _f, _x, _v); \
        f.endEdit(); \
	} \
    template<> void StiffSpringForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) \
    { \
        VecDeriv& _df = *df.beginEdit(); \
        const VecDeriv& _dx = dx.getValue();\
		data.addDForce(this, true, _df, _dx, mparams->kFactor(), sofa::core::mechanicalparams::bFactor(mparams)); \
        df.endEdit(); \
	}

CudaSpringForceField_ImplMethods(gpu::cuda::CudaVec3fTypes);
CudaSpringForceField_ImplMethods(gpu::cuda::CudaVec3f1Types);

#ifdef SOFA_GPU_CUDA_DOUBLE

CudaSpringForceField_ImplMethods(gpu::cuda::CudaVec3dTypes);
CudaSpringForceField_ImplMethods(gpu::cuda::CudaVec3d1Types);

#endif // SOFA_GPU_CUDA_DOUBLE

#undef CudaSpringForceField_ImplMethods

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif
