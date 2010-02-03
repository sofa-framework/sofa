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
#ifndef SOFA_COMPONENT_COLLISION_CUDADIAGONALMASS_INL
#define SOFA_COMPONENT_COLLISION_CUDADIAGONALMASS_INL

#include "CudaDiagonalMass.h"
#include <sofa/component/mass/DiagonalMass.inl>


namespace sofa
{
namespace gpu
{
namespace cuda
{

using namespace sofa::gpu::cuda;
using namespace sofa::component::linearsolver;

extern "C"
{
    void DiagonalMassCuda_addMDxf(unsigned int size, float factor, const void * mass, const void* dx, void* res);
    void DiagonalMassCuda_addMDxd(unsigned int size, double factor, const void * mass, const void* dx, void* res);

    void DiagonalMassCuda_accFromFf(unsigned int size, const void * mass, const void* f, void* a);
    void DiagonalMassCuda_accFromFd(unsigned int size, const void * mass, const void* f, void* a);
}

}

}

namespace component
{

namespace mass
{

template<>
void DiagonalMass<CudaVec3fTypes, float>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor)
{
// 	DiagonalMassCuda_addMDxf(dx.size(),(float) factor, f_mass.getValue().deviceRead() , dx.deviceRead(), res.deviceWrite());
}

template<>
void DiagonalMass<CudaVec3fTypes, float>::accFromF(VecDeriv& a, const VecDeriv& f)
{
// 	DiagonalMassCuda_accFromFf(f.size(),  f_mass.getValue().deviceRead(), f.deviceRead(), a.deviceWrite());
}

// template <>
// void DiagonalMass<CudaVec3fTypes, float>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&) {
//     // weight
//     Vec3d g ( this->getContext()->getLocalGravity() );
// 	Deriv theGravity;
// 	DataTypes::set( theGravity, g[0], g[1], g[2]);
// 	Deriv mg = theGravity * mass.getValue();
// 	UniformMassCuda3f_addForce(f.size(), mg.ptr(), f.deviceWrite());
// }

template<>
bool DiagonalMass<CudaVec3fTypes, float>::addBBox(double* minBBox, double* maxBBox)
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


#ifdef SOFA_GPU_CUDA_DOUBLE

template<>
void DiagonalMass<CudaVec3dTypes, double>::addMDx(VecDeriv& res, const VecDeriv& dx, double factor)
{
// 	DiagonalMassCuda_addMDxd(dx.size(),(double) factor, f_mass.getValue().deviceRead() , dx.deviceRead(), res.deviceWrite());
}

template<>
void DiagonalMass<CudaVec3dTypes, double>::accFromF(VecDeriv& a, const VecDeriv& f)
{
// 	DiagonalMassCuda_accFromFd(f.size(),  f_mass.getValue().deviceRead(), f.deviceRead(), a.deviceWrite());
}

// template<>
// void DiagonalMass<CudaVec3dTypes, double>::addForce(VecDeriv& f, const VecCoord&, const VecDeriv&) {
// //     // weight
// //     Vec3d g ( this->getContext()->getLocalGravity() );
// // 	Deriv theGravity;
// // 	DataTypes::set( theGravity, g[0], g[1], g[2]);
// // 	Deriv mg = theGravity * mass.getValue();
// // 	UniformMassCuda3d_addForce(f.size(), mg.ptr(), f.deviceWrite());
// }

template <>
bool DiagonalMass<CudaVec3dTypes, double>::addBBox(double* minBBox, double* maxBBox)
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

#endif

} // namespace mass

} // namespace component

} // namespace sofa

#endif
