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
#ifndef SOFAOPENCL_OPENCLSPRINGFORCEFIELD_INL
#define SOFAOPENCL_OPENCLSPRINGFORCEFIELD_INL

#include "OpenCLSpringForceField.h"
#define DEBUG_TEXT(t) //printf("\t%s\t %s %d\n",t,__FILE__,__LINE__);

namespace sofa
{

namespace gpu
{

namespace opencl
{

extern "C"
{
    extern void SpringForceFieldOpenCL3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v);
    extern void SpringForceFieldOpenCL3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2);
    extern void StiffSpringForceFieldOpenCL3f_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3f_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3f_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, float factor);
    extern void StiffSpringForceFieldOpenCL3f_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, float factor);

    extern void SpringForceFieldOpenCL3f1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v);
    extern void SpringForceFieldOpenCL3f1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2);
    extern void StiffSpringForceFieldOpenCL3f1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3f1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3f1_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, double factor);
    extern void StiffSpringForceFieldOpenCL3f1_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, double factor);



    extern void SpringForceFieldOpenCL3d_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v);
    extern void SpringForceFieldOpenCL3d_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2);
    extern void StiffSpringForceFieldOpenCL3d_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3d_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3d_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, double factor);
    extern void StiffSpringForceFieldOpenCL3d_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, double factor);

    extern void SpringForceFieldOpenCL3d1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v);
    extern void SpringForceFieldOpenCL3d1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2);
    extern void StiffSpringForceFieldOpenCL3d1_addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3d1_addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx);
    extern void StiffSpringForceFieldOpenCL3d1_addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, double factor);
    extern void StiffSpringForceFieldOpenCL3d1_addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, double factor);

} // extern "C"


template<>
class OpenCLKernelsSpringForceField<OpenCLVec3fTypes>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v)
    {   SpringForceFieldOpenCL3f_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2)
    {   SpringForceFieldOpenCL3f_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3f_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3f_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3f_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, (float)factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3f_addExternalDForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, dx1, x1, offset2, dx2, x2, dfdx, (float)factor); }
};

template<>
class OpenCLKernelsSpringForceField<OpenCLVec3f1Types>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v)
    {   SpringForceFieldOpenCL3f1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2)
    {   SpringForceFieldOpenCL3f1_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3f1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3f1_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3f1_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3f1_addExternalDForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, dx1, x1, offset2, dx2, x2, dfdx, factor); }
};



template<>
class OpenCLKernelsSpringForceField<OpenCLVec3dTypes>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v)
    {   SpringForceFieldOpenCL3d_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2)
    {   SpringForceFieldOpenCL3d_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3d_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3d_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3d_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3d_addExternalDForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, dx1, x1, offset2, dx2, x2, dfdx, factor); }
};

template<>
class OpenCLKernelsSpringForceField<OpenCLVec3d1Types>
{
public:
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v)
    {   SpringForceFieldOpenCL3d1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2)
    {   SpringForceFieldOpenCL3d1_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2); }
    static void addForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer x, const _device_pointer v, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3d1_addForce(nbVertex, nbSpringPerVertex, springs, f, x, v, dfdx); }
    static void addExternalForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer x1, const _device_pointer v1, unsigned int offset2, const _device_pointer x2, const _device_pointer v2, _device_pointer dfdx)
    {   StiffSpringForceFieldOpenCL3d1_addExternalForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, x1, v1, offset2, x2, v2, dfdx); }
    static void addDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, _device_pointer f, const _device_pointer dx, const _device_pointer x, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3d1_addDForce(nbVertex, nbSpringPerVertex, springs, f, dx, x, dfdx, factor); }
    static void addExternalDForce(unsigned int nbVertex, unsigned int nbSpringPerVertex, const _device_pointer springs, unsigned int offset1, _device_pointer f1, const _device_pointer dx1, const _device_pointer x1, unsigned int offset2, const _device_pointer dx2, const _device_pointer x2, const _device_pointer dfdx, SReal factor)
    {   StiffSpringForceFieldOpenCL3d1_addExternalDForce(nbVertex, nbSpringPerVertex, springs, offset1, f1, dx1, x1, offset2, dx2, x2, dfdx, factor); }
};



} // namespace opencl

} // namespace gpu

namespace component
{

namespace interactionforcefield
{


using namespace gpu::opencl;

template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::init(Main* m, bool stiff)
{
    DEBUG_TEXT("SpringForceFieldInternalData::init");
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
            /// TODO: OpenCL currently does not support arbitrary offsets into buffers, force the first vertex to be 0
            //nsprings1[0] += 0;
            //nsprings2[0] += 0;
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
            /// TODO: OpenCL currently does not support arbitrary offsets into buffers, force the first vertex to be 0
            //nsprings[0] += 0;

            int nmax = 0;
            for (std::map<int,int>::const_iterator it = nsprings.begin(); it != nsprings.end(); ++it)
                if (it->second > nmax)
                    nmax = it->second;
            data.springs1.init(nsprings.begin()->first, nsprings.rbegin()->first - nsprings.begin()->first + 1, nmax);
            std::cout << "OpenCL SpringForceField: "<<springs.size()<<" springs, "<<data.springs1.nbVertex<<" attached points, max "<<data.springs1.nbSpringPerVertex<<" springs per point."<<std::endl;
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
        std::cout <<__LINE__ << __FILE__ << "\ndata.spring1.dfdx " << data.springs1.springs.size() << "\n";
        data.springs1.dfdx.resize(data.springs1.springs.size());
        data.springs2.dfdx.resize(data.springs2.springs.size());
    }
    DEBUG_TEXT("END: SpringForceFieldInternalData::init");
}

// -- InteractionForceField interface
template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::addForce(Main* m, bool stiff, VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2)
{
    DEBUG_TEXT("SpringForceFieldInternalData::addDForce");
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
                        OpenCLMemoryManager<Deriv>::deviceOffset(f.deviceWrite(),d),
                        OpenCLMemoryManager<Coord>::deviceOffset(x.deviceRead(),d),
                        OpenCLMemoryManager<Deriv>::deviceOffset(v.deviceRead(),d));
            else
                Kernels::addForce(data.springs1.nbVertex,
                        data.springs1.nbSpringPerVertex,
                        data.springs1.springs.deviceRead(),
                        OpenCLMemoryManager<Deriv>::deviceOffset(f.deviceWrite(),d),
                        OpenCLMemoryManager<Coord>::deviceOffset(x.deviceRead(),d),
                        OpenCLMemoryManager<Deriv>::deviceOffset(v.deviceRead(),d),
                        data.springs1.dfdx.deviceWrite());

        }
    }
    else
    {
        f1.resize(x1.size());
        f2.resize(x2.size());
        if (data.springs1.nbSpringPerVertex > 0)
        {
            if (!stiff)
                Kernels::addExternalForce(data.springs1.nbVertex,
                        data.springs1.nbSpringPerVertex,
                        data.springs1.springs.deviceRead(),
                        data.springs1.vertex0,
                        f1.deviceWrite(),
                        x1.deviceRead(),
                        v1.deviceRead(),
                        data.springs2.vertex0,
                        x2.deviceRead(),
                        v2.deviceRead());
            else
                Kernels::addExternalForce(data.springs1.nbVertex,
                        data.springs1.nbSpringPerVertex,
                        data.springs1.springs.deviceRead(),
                        data.springs1.vertex0,
                        f1.deviceWrite(),
                        x1.deviceRead(),
                        v1.deviceRead(),
                        data.springs2.vertex0,
                        x2.deviceRead(),
                        v2.deviceRead(),
                        data.springs1.dfdx.deviceWrite());
        }
        if (data.springs2.nbSpringPerVertex > 0)
        {
            if (!stiff)
                Kernels::addExternalForce(data.springs2.nbVertex,
                        data.springs2.nbSpringPerVertex,
                        data.springs2.springs.deviceRead(),
                        data.springs2.vertex0,
                        f2.deviceWrite(),
                        x2.deviceRead(),
                        v2.deviceRead(),
                        data.springs1.vertex0,
                        x1.deviceRead(),
                        v1.deviceRead());
            else
                Kernels::addExternalForce(data.springs2.nbVertex,
                        data.springs2.nbSpringPerVertex,
                        data.springs2.springs.deviceRead(),
                        data.springs2.vertex0,
                        f2.deviceWrite(),
                        x2.deviceRead(),
                        v2.deviceRead(),
                        data.springs1.vertex0,
                        x1.deviceRead(),
                        v1.deviceRead(),
                        data.springs2.dfdx.deviceWrite());
        }
    }
    DEBUG_TEXT("END: SpringForceFieldInternalData::addDForce");
}

template<class TCoord, class TDeriv, class TReal>
void SpringForceFieldInternalData< gpu::opencl::OpenCLVectorTypes<TCoord,TDeriv,TReal> >::addDForce(Main* m, bool stiff, VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, SReal kFactor, SReal /*bFactor*/)
{
    DEBUG_TEXT("SpringForceFieldInternalData::addDForce");

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
                    OpenCLMemoryManager<Deriv>::deviceOffset(df.deviceWrite() , d),
                    OpenCLMemoryManager<Deriv>::deviceOffset(dx.deviceRead() , d),
                    OpenCLMemoryManager<Coord>::deviceOffset(x.deviceRead()  , d),
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
        if (data.springs1.nbSpringPerVertex > 0)
        {
            Kernels::addExternalDForce(data.springs1.nbVertex,
                    data.springs1.nbSpringPerVertex,
                    data.springs1.springs.deviceRead(),
                    data.springs1.vertex0,
                    df1.deviceWrite(),
                    x1.deviceRead(),
                    dx1.deviceRead(),
                    data.springs2.vertex0,
                    x2.deviceRead(),
                    dx2.deviceRead(),
                    data.springs1.dfdx.deviceRead(),
                    kFactor);
        }
        if (data.springs2.nbSpringPerVertex > 0)
        {
            Kernels::addExternalDForce(data.springs2.nbVertex,
                    data.springs2.nbSpringPerVertex,
                    data.springs2.springs.deviceRead(),
                    data.springs2.vertex0,
                    df2.deviceWrite(),
                    dx2.deviceRead(),
                    x2.deviceRead(),
                    data.springs1.vertex0,
                    dx1.deviceRead(),
                    x1.deviceRead(),
                    data.springs2.dfdx.deviceRead(),
                    kFactor);
        }
    }
    DEBUG_TEXT("END: SpringForceFieldInternalData::addDForce");
}


// I know using macros is bad design but this is the only way not to repeat the code for all OpenCL types
#define OpenCLSpringForceField_ImplMethods(T) \
	template<> void SpringForceField< T >::init() \
	{data.init(this, false); } \
    template<> void SpringForceField< T >::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2) \
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
    { data.init(this, true); } \
    template<> void StiffSpringForceField< T >::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f1, DataVecDeriv& d_f2, const DataVecCoord& d_x1, const DataVecCoord& d_x2, const DataVecDeriv& d_v1, const DataVecDeriv& d_v2) \
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
    template<> void StiffSpringForceField< T >::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df1, DataVecDeriv& d_df2, const DataVecDeriv& d_dx1, const DataVecDeriv& d_dx2) \
    { \
		VecDeriv& df1 = *d_df1.beginEdit(); \
		const VecDeriv& dx1 = d_dx1.getValue(); \
		VecDeriv& df2 = *d_df2.beginEdit(); \
		const VecDeriv& dx2 = d_dx2.getValue(); \
		data.addDForce(this, true, df1, df2, dx1, dx2, mparams->kFactor(), mparams->bFactor()); \
		d_df1.endEdit(); \
		d_df2.endEdit(); \
	}


OpenCLSpringForceField_ImplMethods(gpu::opencl::OpenCLVec3fTypes)
OpenCLSpringForceField_ImplMethods(gpu::opencl::OpenCLVec3f1Types)
OpenCLSpringForceField_ImplMethods(gpu::opencl::OpenCLVec3dTypes)
OpenCLSpringForceField_ImplMethods(gpu::opencl::OpenCLVec3d1Types)

//#undef OpenCLSpringForceField_ImplMethods

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#undef DEBUG_TEXT

#endif
