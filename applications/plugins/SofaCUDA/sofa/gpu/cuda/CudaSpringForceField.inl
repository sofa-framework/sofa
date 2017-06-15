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
#ifndef SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDASPRINGFORCEFIELD_INL

#include "CudaSpringForceField.h"
#include <SofaDeformable/SpringForceField.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <SofaDeformable/MeshSpringForceField.inl>
#include <SofaGeneralDeformable/TriangleBendingSprings.inl>
#include <SofaGeneralDeformable/QuadBendingSprings.inl>

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
    const sofa::helper::vector<Spring>& springs = m->springs.getValue();
    if (!springs.empty())
    {
        bool external = (m->mstate1!=m->mstate2);
        if (external)
        {
            std::map<int,int> nsprings1;
            std::map<int,int> nsprings2;
            for (unsigned int i=0; i<springs.size(); i++)
            {
                nsprings1[springs[i].m1]++;
                nsprings2[springs[i].m2]++;
            }

            int nmax1 = 0;
            for (std::map<int,int>::const_iterator it = nsprings1.begin(); it != nsprings1.end(); ++it)
                if (it->second > nmax1)
                    nmax1 = it->second;
            data.springs1.init(nsprings1.begin()->first, nsprings1.rbegin()->first - nsprings1.begin()->first + 1, nmax1);

            int nmax2 = 0;
            for (std::map<int,int>::const_iterator it = nsprings2.begin(); it != nsprings2.end(); ++it)
                if (it->second > nmax2)
                    nmax2 = it->second;
            data.springs2.init(nsprings2.begin()->first, nsprings2.rbegin()->first - nsprings2.begin()->first + 1, nmax2);

            nsprings1.clear();
            nsprings2.clear();
            for (unsigned int i=0; i<springs.size(); i++)
            {
                int m1 = springs[i].m1 - data.springs1.vertex0;
                int m2 = springs[i].m2 - data.springs2.vertex0;
                data.springs1.set(m1, nsprings1[m1]++, m2,
                        (float)springs[i].initpos,
                        (float)springs[i].ks,
                        (float)springs[i].kd);
                data.springs2.set(m2, nsprings2[m2]++, m1,
                        (float)springs[i].initpos,
                        (float)springs[i].ks,
                        (float)springs[i].kd);
            }
        }
        else
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
    }
    if (stiff)
    {
        data.springs1.dfdx.resize(data.springs1.springs.size());
        data.springs2.dfdx.resize(data.springs2.springs.size());
    }
}

// -- InteractionForceField interface
template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, bool stiff, VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    Data& data = m->data;

    if (m->mstate1 == m->mstate2)
    {
        VecDeriv& f = f1;
        const VecCoord& x = x1;
        const VecDeriv& v = v1;
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
    else
    {
        f1.resize(x1.size());
        f2.resize(x2.size());
        int d1 = data.springs1.vertex0;
        int d2 = data.springs2.vertex0;
        if (data.springs1.nbSpringPerVertex > 0)
        {
            if (!stiff)
                Kernels::addExternalForce(data.springs1.nbVertex,
                        data.springs1.nbSpringPerVertex,
                        data.springs1.springs.deviceRead(),
                        (      Deriv*)f1.deviceWrite() + d1,
                        (const Coord*)x1.deviceRead()  + d1,
                        (const Deriv*)v1.deviceRead()  + d1,
                        (const Coord*)x2.deviceRead()  + d2,
                        (const Deriv*)v2.deviceRead()  + d2);
            else
                Kernels::addExternalForce(data.springs1.nbVertex,
                        data.springs1.nbSpringPerVertex,
                        data.springs1.springs.deviceRead(),
                        (      Deriv*)f1.deviceWrite() + d1,
                        (const Coord*)x1.deviceRead()  + d1,
                        (const Deriv*)v1.deviceRead()  + d1,
                        (const Coord*)x2.deviceRead()  + d2,
                        (const Deriv*)v2.deviceRead()  + d2,
                        data.springs1.dfdx.deviceWrite());
        }
        if (data.springs2.nbSpringPerVertex > 0)
        {
            if (!stiff)
                Kernels::addExternalForce(data.springs2.nbVertex,
                        data.springs2.nbSpringPerVertex,
                        data.springs2.springs.deviceRead(),
                        (      Deriv*)f2.deviceWrite() + d2,
                        (const Coord*)x2.deviceRead()  + d2,
                        (const Deriv*)v2.deviceRead()  + d2,
                        (const Coord*)x1.deviceRead()  + d1,
                        (const Deriv*)v1.deviceRead()  + d1);
            else
                Kernels::addExternalForce(data.springs2.nbVertex,
                        data.springs2.nbSpringPerVertex,
                        data.springs2.springs.deviceRead(),
                        (      Deriv*)f2.deviceWrite() + d2,
                        (const Coord*)x2.deviceRead()  + d2,
                        (const Deriv*)v2.deviceRead()  + d2,
                        (const Coord*)x1.deviceRead()  + d1,
                        (const Deriv*)v1.deviceRead()  + d1,
                        data.springs2.dfdx.deviceWrite());
        }
    }
}

template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> >::addDForce(Main* m, bool stiff, VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, SReal kFactor, SReal /*bFactor*/)
{
    if (!stiff) return;
    Data& data = m->data;
    if (m->mstate1 == m->mstate2)
    {
        VecDeriv& df = df1;
        const VecDeriv& dx = dx1;
        const VecCoord& x = m->mstate1->read(core::ConstVecCoordId::position())->getValue();
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
    else
    {
        const VecCoord& x1 = m->mstate1->read(core::ConstVecCoordId::position())->getValue();
        const VecCoord& x2 = m->mstate2->read(core::ConstVecCoordId::position())->getValue();
        df1.resize(x1.size());
        df2.resize(x2.size());
        int d1 = data.springs1.vertex0;
        int d2 = data.springs2.vertex0;
        if (data.springs1.nbSpringPerVertex > 0)
        {
            Kernels::addExternalDForce(data.springs1.nbVertex,
                    data.springs1.nbSpringPerVertex,
                    data.springs1.springs.deviceRead(),
                    (      Deriv*)df1.deviceWrite() + d1,
                    (const Coord*)x1.deviceRead()  + d1,
                    (const Deriv*)dx1.deviceRead()  + d1,
                    (const Coord*)x2.deviceRead()  + d2,
                    (const Deriv*)dx2.deviceRead()  + d2,
                    data.springs1.dfdx.deviceRead(),
                    kFactor);
        }
        if (data.springs2.nbSpringPerVertex > 0)
        {
            Kernels::addExternalDForce(data.springs2.nbVertex,
                    data.springs2.nbSpringPerVertex,
                    data.springs2.springs.deviceRead(),
                    (      Deriv*)df2.deviceWrite() + d2,
                    (const Deriv*)dx2.deviceRead() + d2,
                    (const Coord*)x2.deviceRead()  + d2,
                    (const Deriv*)dx1.deviceRead() + d1,
                    (const Coord*)x1.deviceRead()  + d1,
                    data.springs2.dfdx.deviceRead(),
                    kFactor);
        }
    }
}


// I know using macros is bad design but this is the only way not to repeat the code for all CUDA types
#define CudaSpringForceField_ImplMethods(T) \
    template<> void SpringForceField< T >::init() \
    { \
	    this->PairInteractionForceField< T >::init();   \
        data.init(this, false); \
    } \
    template<> void SpringForceField< T >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2) \
    { \
		VecDeriv& f1 = *d_f1.beginEdit(); \
		const VecCoord& x1 = d_x1.getValue(); \
		const VecDeriv& v1 = d_v1.getValue(); \
		VecDeriv& f2 = *d_f2.beginEdit(); \
		const VecCoord& x2 = d_x2.getValue(); \
		const VecDeriv& v2 = d_v2.getValue(); \
		data.addForce(this, false, f1, f2, x1, x2, v1, v2);\
		d_f1.endEdit(); \
		d_f2.endEdit(); \
	} \
    template<> void StiffSpringForceField< T >::init() \
    { \
	    this->PairInteractionForceField< T >::init(); \
        data.init(this, true); \
    } \
    template<> void StiffSpringForceField< T >::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2) \
    { \
		VecDeriv& f1 = *d_f1.beginEdit(); \
		const VecCoord& x1 = d_x1.getValue(); \
		const VecDeriv& v1 = d_v1.getValue(); \
		VecDeriv& f2 = *d_f2.beginEdit(); \
		const VecCoord& x2 = d_x2.getValue(); \
		const VecDeriv& v2 = d_v2.getValue(); \
		data.addForce(this, true, f1, f2, x1, x2, v1, v2); \
		d_f1.endEdit(); \
		d_f2.endEdit(); \
	} \
    template<> void StiffSpringForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df1, DataVecDeriv& d_df2, const DataVecDeriv& d_dx1, const DataVecDeriv& d_dx2) \
    { \
		VecDeriv& df1 = *d_df1.beginEdit(); \
		const VecDeriv& dx1 = d_dx1.getValue(); \
		VecDeriv& df2 = *d_df2.beginEdit(); \
		const VecDeriv& dx2 = d_dx2.getValue(); \
		data.addDForce(this, true, df1, df2, dx1, dx2, mparams->kFactor(), mparams->bFactor()); \
		d_df1.endEdit(); \
		d_df2.endEdit(); \
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
